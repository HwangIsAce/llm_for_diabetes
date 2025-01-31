import logging
import os
import ast
import time
import pandas as pd
import torch
import random
import numpy as np
import faiss
from tqdm import tqdm

from datasets import (load_dataset, 
                      Dataset, 
                      DatasetDict,
                      concatenate_datasets)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dataset_split(dataset: Dataset) -> Dataset:
    dataset = dataset['train'].to_pandas()
    dataset = dataset(columns=["split"])

    train_df, temp_df = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=True, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    test_dataset = Dataset.from_pandas(test_df)

    new_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    })

    return new_dataset

def contains_keywords(text, keywords):
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def process_dataset_split(dataset_name, split_name, dataset, text_columns, output_file, keywords=["diabetes", "insulin"]):
    rows = []
    for i in range(len(dataset)):
        row = dataset[i]

        for col in text_columns:
            if col in row:
                text = row[col]

            elif col == "abstract" and "passages" in row:
                if isinstance(row["passages"], list):
                    text = " ".join([p["text"] for p in row["passages"] if p.get("type") == "abstract"])
                else:
                    text = ""
            else:
                text = ""

            if not text:
                continue

            is_related = 1 if contains_keywords(text, keywords) else 0

            rows.append({
                "dataset": dataset_name,
                "split_data": split_name,
                "features": row,  
                "input": text,
                "output": is_related
            })
    if rows:  
        df = pd.DataFrame(rows)
        df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
        logging.info(f"{dataset_name} ({split_name}): {len(rows)} rows processed and appended to {output_file}")
    else:
        logging.info(f"{dataset_name} ({split_name}): No rows processed.")

def re_train_test_split(dataframe, train_ratio=0.8): 
    new_rows = []

    grouped = dataframe.groupby('dataset')

    for dataset_name, group in grouped:
        if dataset_name not in ["medqa", "medmcqa", "pubmedqa", "bionli", "chemdner", "ddi", "meddialog", "pubmed"]:
            continue

        group = group.drop_duplicates(subset="features")

        train_data, test_data = train_test_split(
            group, test_size=1-train_ratio, random_state=42,
        )

        train_data = train_data.copy()
        train_data["split_data"] = 'train'

        test_data = test_data.copy()
        test_data["split_data"] = 'test'

        new_rows.append(train_data)
        new_rows.append(test_data)

    result_df = pd.concat(new_rows, ignore_index=True)
    return result_df

def extract_for_sampling(row):
    features = ast.literal_eval(row['features'])

    if row['dataset']=='medqa':    
        return {'input': features.get('question', None), 'output': features.get('answer', None)}
    elif row['dataset']=='medmcqa':
        answer = ", ".join([features.get('opa', ''), features.get('opb', ''), features.get('opc', ''), features.get('opd', ''),
        ]).strip() 
        return {'input': features.get('question', None), 'output': answer}
    elif row['dataset']=='pubmedqa':
        return {'input': features.get('QUESTION', None), 'output': features.get('LONG_ANSWER', None)}
    elif row['dataset']=='bionli':
        query = features.get('query', '') 
        input = query.split("INPUT: ")[1].split("[HYP]")[0].strip() if "INPUT: " in query else None
        return {'input': input, 'output': features.get('answer', None)}
    elif row['dataset']=='chemdner':
        answer = features.get('entities', None)
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0]
        else:
            answer = ""
        return {'input': features.get('text', None), 'output': answer}
    elif row['dataset']=='ddi':
        input = features.get('conversations', [])[0].get('value', None).split("INPUT:")[1].split("OUTPUT:")[0].strip()
        answer = features.get('conversations', [])[1].get('value', None)
        return {'input': input, 'output': answer}
    elif row['dataset']=='meddialog':
        input = features.get('src', None).split("Doctor:")[0].replace("Patient:", "").strip()
        answer = features.get('src', None).split("Doctor:")[1].strip()
        return {'input': input, 'output': answer}
    elif row['dataset']=='pubmed':
        return {'input': features.get('article', None), 'output': features.get('abstract', None)}
    
    return {'input': None, 'output': None}

def balance_chemdner(dataframe):
    chemdner_df = dataframe[dataframe['dataset'] == 'chemdner'].copy()
    
    empty_outputs = chemdner_df[chemdner_df['for_sampling'].apply(lambda x: x['output'] == '')]
    non_empty_outputs = chemdner_df[chemdner_df['for_sampling'].apply(lambda x: x['output'] != '')]
    
    target_empty_size = int(len(non_empty_outputs) * 0.05 / 0.95) 
    sampled_empty_outputs = empty_outputs.sample(n=min(target_empty_size, len(empty_outputs)), random_state=42)
    
    balanced_chemdner_df = pd.concat([non_empty_outputs, sampled_empty_outputs])
    
    other_datasets = dataframe[dataframe['dataset'] != 'chemdner']
    final_dataframe = pd.concat([other_datasets, balanced_chemdner_df])
    
    return final_dataframe

def mmr(query_embedding, doc_embeddings, diversity, top_n):
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    selected_indices = []
    candidate_indices = list(range(len(doc_embeddings)))

    for _ in range(top_n):
        if not candidate_indices:
            break

        if len(selected_indices) == 0:
            selected_idx = candidate_indices[np.argmax(np.dot(doc_embeddings[candidate_indices], query_embedding.T))]
        else:
            selected_embeddings = doc_embeddings[selected_indices]
            similarity_to_selected = np.dot(doc_embeddings[candidate_indices], selected_embeddings.T)
            diversity_scores = np.max(similarity_to_selected, axis=1)
            relevance_scores = np.dot(doc_embeddings[candidate_indices], query_embedding.T).flatten()
            mmr_scores = (1 - diversity) * relevance_scores - diversity * diversity_scores
            selected_idx = candidate_indices[np.argmax(mmr_scores)]

        selected_indices.append(selected_idx)
        candidate_indices.remove(selected_idx)

    return selected_indices

def mmr_sampling(dataframe, sampling_dict, embedding_model='all-MiniLM-L6-v2', batch_size=64, diversity=0.7, seed=42):
    set_seed(seed)

    print("Loading SentenceTransformer model on GPU...")
    model = SentenceTransformer(embedding_model, device="cuda:0")

    sampled_rows = []

    for dataset_name, sample_count in tqdm(sampling_dict.items(), desc="Processing datasets"):
        subset = dataframe[(dataframe['dataset'] == dataset_name) & (dataframe['split_data'] == 'train')].copy()
        num_rows = len(subset)
        if num_rows == 0:
            print(f"No data found for dataset: {dataset_name}")
            continue

        print(f"Generating embeddings for dataset: {dataset_name} ({num_rows} rows)")
        start_time = time.time()

        embeddings = model.encode(
            subset['for_sampling'].astype(str).tolist(),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        elapsed_time = time.time() - start_time
        print(f"Embedding generation took {elapsed_time:.2f} seconds for {num_rows} rows")

        query_embedding = np.mean(embeddings, axis=0)

        print(f"Applying MMR for dataset: {dataset_name}")
        selected_indices = mmr(query_embedding, embeddings, diversity, sample_count)
        sampled_subset = subset.iloc[selected_indices]
        sampled_rows.append(sampled_subset)

    return pd.concat(sampled_rows, ignore_index=True)

def parse_features(row):
    try:
        return ast.literal_eval(row)
    except (ValueError, SyntaxError):
        return {}
    
def generate_instruction_input_output(row):
    instruction_value = ""; input_value = ""; output_value =""
    
    features = parse_features(row['features'])

    task = row['task']
    dataset = row['dataset']

    if dataset=='medqa':
        question = features.get('question', None)
        options = features.get('options', None)
        answer = features.get('answer', None)
        answer_idx = features.get('answer_idx', None)

        instruction_value = "Select the most appropriate answer for the given medical question from the provided options."
        input_value = (
            f"{question} Please select one of the following: A) {options['A']}, B) {options['B']}, C) {options['C']}, D) {options['D']}."
        )
        output_value = (
            f"{answer_idx}) {answer}"
        )
    elif dataset=='medmcqa':
        question = features.get('question', None)
        options = {
            'A': features.get('opa', '').strip(),
            'B': features.get('opb', '').strip(),
            'C': features.get('opc', '').strip(),
            'D': features.get('opd', '').strip(),
        }
        answer_num = features.get('cop', None)
        answer_idx = 'A' if answer_num == 0 else 'B' if answer_num == 1 else 'C' if answer_num == 2 else 'D'
        answer = features.get('opa', '').strip() if answer_num == 0 else features.get('opb', '').strip() if answer_num == 1 else features.get('opc', '').strip() if answer_num == 2 else features.get('opd', '').strip()

        instruction_value = "Select the most appropriate answer for the given medical question from the provided options."
        input_value = (
            f"{question} Please select one of the following: A) {options['A']}, B) {options['B']}, C) {options['C']}, D) {options['D']}."
        )
        output_value = (
            f"{answer_idx}) {answer}"
        )
    elif dataset=='pubmedqa':
        question = features.get('QUESTION', '').strip()
        context = " ".join(features.get('CONTEXTS'))
        # answer = features.get('LONG_ANSWER', '').strip() 
        answer = features.get('final_decision', '').strip()
        
        instruction_value = "Choose the correct anser (Yes, No, or Maybe) for the given question based on the proviced context."
        input_value = (
            f"Question: {question} "
            f"Context: {context}"
        )
        output_value = f"{answer}"
    elif dataset=='bionli':
        query = features.get('query', '') 
        
        instruction_value =  "Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. return only the label."
        input_value = query.split("INPUT: ")[1].split("[HYP]")[0].strip() if "INPUT: " in query else None
        output_value = features.get('answer', None)
    elif dataset=='chemdner':
        query = features.get('query', '') 
        
        instruction_value =  "Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. return only the label."
        input_value = query.split("INPUT: ")[1].split("[HYP]")[0].strip() if "INPUT: " in query else None
        output_value = features.get('answer', None)
    elif row['dataset']=='ddi':
        instruction_value = "Analyze the sentence with two drugs labeled as @DRUG_A$ and @DRUG_B$. Extract the interaction between @DRUG_A$ and @DRUG_B$ from the input sentence by selecting only one of the following options: 'DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int'. 'DDI-effect': Choose this when the interaction describes an effect or a pharmacodynamic mechanism. 'DDI-mechanism': Choose this for interactions explained by pharmacokinetic mechanisms. 'DDI-advise': Choose this when the sentence provides a recommendation or advice about the drug interaction. 'DDI-false': Choose this if there is no actual drug-drug interaction in the sentence. 'DDI-int': Choose this when a drug-drug interaction is mentioned without additional detail."
        input_value = features.get('conversations', [])[0].get('value', None).split("INPUT:")[1].split("OUTPUT:")[0].strip()
        output_value = features.get('conversations', [])[1].get('value', None)
    elif row['dataset']=='meddialog':
        instruction_value = features.get('tgt', None)
        input_value = features.get('src', None).split("Doctor:")[0].replace("Patient:", "").strip()
        output_value = features.get('src', None).split("Doctor:")[1].strip()
    elif row['dataset']=='pubmed':
        article = features.get('article', '').strip()  
        abstract = features.get('abstract', '').strip()

        instruction_value = "Summarize the key findings of the given PubMed abstract into structured fields: Objective, Methods, Results, and Conclusion."
        input_value = f"{article}"
        output_value = abstract
    
    return instruction_value, input_value, output_value

def remove_imbalance(dataframe, dataset_name, class_name_1, class_name_2): # in 2 classes
    df_train = dataframe[dataframe['dataset'] == dataset_name]
    df_1 = df_train[df_train['output']== class_name_1]
    df_2 = df_train[df_train['output']== class_name_2]

    df_larger = df_1 if len(df_1) > len(df_2) else df_2
    df_smaller = df_1 if len(df_1) < len(df_2) else df_2

    df_downsample = resample(df_larger, replace=False, n_samples=len(df_smaller))
    balanced_df = pd.concat([df_downsampled, df_smaller])

    final_df_balanced = dataframe[dataframe['dataset'] != dataset_name]
    dataframe = pd.concat([final_df_balanced, balance_df])
    return dataframe

if __name__ == "__main__":
    # logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # loading datasets
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options")
    medmcqa = load_dataset("openlifescienceai/medmcqa")
    pubmedqa = load_dataset("bigbio/pubmed_qa", trust_remote_code=True)
    bionli = load_dataset("clinicalnlplab/BioNLI_test")
    ddi = load_dataset("YufeiHFUT/DDI2013")
    chemdner =load_dataset("kjappelbaum/chemnlp-chemdner")
    healthcaremagic = load_dataset("lighteval/med_dialog", "healthcaremagic")
    icliniq = load_dataset("lighteval/med_dialog", "icliniq")
    pubmed = load_dataset("ccdv/pubmed-summarization")

    # preprocessing datasets
    chemdner = dataset_split(chemdner)
    meddialog = {
        "train": concatenate_datasets([healthcaremagic['train'], icliniq['train']]),
        "validation": concatenate_datasets([healthcaremagic['validation'], icliniq['validation']]),
        "test": concatenate_datasets([healthcaremagic['test'], icliniq['test']]),
    }

    # logging datasets shape
    logging.info(medqa)
    logging.info(medmcqa)
    logging.info(pubmedqa)
    logging.info(bionli)
    logging.info(ddi)
    logging.info(chemdner)
    logging.info(meddialog)
    logging.info(pubmed)

    # filtering diabetes-related data
    keywords = [keyword.lower() for keyword in [
    "Diabetes",
    "Diabetes Mellitus", "Diabetes Mellitus, Experimental", "Diabetes Mellitus, Type 1",
    "Wolfram Syndrome", "Diabetes Mellitus, Type 2", "Diabetes Mellitus, Lipoatrophic",
    "Diabetes, Gestational", "Donohue Syndrome", "Latent Autoimmune Diabetes in Adults",
    "Prediabetic State", "Diabetes Complications", "Diabesity", "Diabetic Angiopathies",
    "Diabetic Cardiomyopathies", "Diabetic Coma", "Diabetic Ketoacidosis",
    "Diabetic Nephropathies", "Diabetic Neuropathies", "Fetal Macrosomia"
    ]]

    datasets = {
        "medqa": {"splits": medqa, "columns": ["question", "answer"]},
        "medmcqa": {"splits": medmcqa, "columns": ["question"]},
        "pubmedqa": {"splits": pubmedqa, "columns": ["QUESTION", "CONTEXTS", "LONG_ANSWER"]},
        "bionli": {"splits": bionli, "columns": ["query", "answer"]},
        "ddi": {"splits": ddi, "columns": ["text"]},
        "chemdner": {"splits": chemdner, "columns": ['text']},
        "meddialog": {"splits": meddialog, "columns": ["tgt", "src"]},
        "pubmed": {"splits": pubmed, "columns": ["abstract"]},
    }
    
    output_file = os.path.join("filtered_by_keywords.csv")

    for dataset_name, details in datasets.items():
        splits = details["splits"]
        text_columns = details["columns"]

        for split_name, split_data in splits.items():
            process_dataset_split(dataset_name, split_name, split_data, text_columns, output_file, keywords)

    # loading a filtered total csv file
    temp = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data1_diabetes/filtered_by_keywords.csv")
    diabetes = temp[temp['output']==1]

    # reseperating train/test datasets
    diabetes = re_train_test_split(diabetes, train_ratio=0.85)

    # sampling using MMR search algorithm
    diabetes['for_sampling'] = diabetes.apply(extract_for_sampling, axis=1)
    diabetes = balance_chemdner(diabetes)

    sampling_train_config = {
        'medqa': 1000,
        'medmcqa': 1000,
        'pubmedqa': 8000, # 1000
        'bionli': 1800, # 1500
        'chemdner': 600,
        'ddi': 900,
        'meddialog': 4000,
        'pubmed': 4000,
    }

    sampling_test_config = {
    'medqa': 150,
    'medmcqa': 150,
    'pubmedqa': 1200, # 150 
    'bionli': 270, # 225
    'chemdner': 90,
    'ddi': 135,
    'meddialog': 600,
    'pubmed': 600,
    }

    set_seed(42)

    sampled_train = mmr_sampling(diabetes, sampling_train_config, batch_size=4096, diversity=0.7, seed=42)
    sampled_test = mmr_sampling(diabetes, sampling_test_config, batch_size=4096, diversity=0.7, seed=42)
    
    sampled_train.to_csv("final_combined_train_sample.csv")
    sampled_test.to_csv("final_combined_test_sample.csv")

    # loading a csv file filtered based on keywords and MMR search
    final_combined_train_sample = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data1_diabetes/final_combined_train_sample.csv")
    final_combined_test_sample = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data1_diabetes/final_combined_test_sample.csv")

    # creating task columns

    final_combined_train_sample['task'] = None
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'medqa', 'task'] = 'qa_objective_1'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'medmcqa', 'task'] = 'qa_objective_2'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'pubmedqa', 'task'] = 'qa_objective_3'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'bionli', 'task'] = 'nli'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'chemdner', 'task'] = 'ie'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'ddi', 'task'] = 're'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'meddialog', 'task'] = 'generation'
    final_combined_train_sample.loc[final_combined_train_sample['dataset'] == 'pubmed', 'task'] = 'summarization'

    final_combined_test_sample['task'] = None
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'medqa', 'task'] = 'qa_objective_1'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'medmcqa', 'task'] = 'qa_objective_2'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'pubmedqa', 'task'] = 'qa_objective_3'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'bionli', 'task'] = 'nli'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'chemdner', 'task'] = 'ie'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'ddi', 'task'] = 're'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'meddialog', 'task'] = 'generation'
    final_combined_test_sample.loc[final_combined_test_sample['dataset'] == 'pubmed', 'task'] = 'summarization'

    # creating instruction, input, output columns
    final_combined_train_sample[['instruction', 'input', 'output']] = final_combined_train_sample.apply(lambda row: pd.Series(generate_instruction_input_output(row)), axis=1)

    final_combined_test_sample[['instruction', 'input', 'output']] = final_combined_test_sample.apply(lambda row: pd.Series(generate_instruction_input_output(row)), axis=1)

    # removing imbalance
    final_combined_train_sample = remove_imbalance(final_combined_train_sample, 'pubmedqa', 'yes', 'no')
    final_combined_test_sample = remove_imbalance(final_combined_test_sample, 'pubmedqa', 'yes', 'no')

    final_combined_train_sample = remove_imbalance(final_combined_train_sample, 'bionli', 'contradiction', 'entailment')
    final_combined_test_sample = remove_imbalance(final_combined_test_sample, 'bionli', 'contradiction', 'entailment')

    # final dataframe
    final_df = pd.concat([final_combined_train_sample, final_combined_test_sample])

    final_combined_train_sample.to_json("/data/jaesung/llm_for_diabetes/src/data/data1_diabetes/train_instruction_dataset.json", orient="columns", indent=4)
    final_combined_test_sample.to_json("/data/jaesung/llm_for_diabetes/src/data/data1_diabetes/test_instruction_dataset.json", orient="columns", indent=4)