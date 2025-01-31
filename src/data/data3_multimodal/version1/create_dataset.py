import pandas as pd
import ast
import logging
import json
from tqdm import tqdm
import os
from difflib import SequenceMatcher

import openai
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split

def extract_food_description(row):
    try:
        row = row.replace("nan", "None")
        data = ast.literal_eval(row)
        return data.get('Food_Description', None) 
    except (ValueError, SyntaxError, TypeError) as e:
        logging.info(f"Error parsing row: {row}, Error: {e}")
        return None 
    
def extract_context_info(context, title):
    try:
        entries = context.split("\n")
        for entry in entries:
            if f"Title: {title}," in entry:
                description = entry.split("Description:")[1].split(", Nutrition Facts:")[0].strip()
                nutrition_facts_str = entry.split("Nutrition Facts:")[1].split(", Ingredients:")[0].strip()
                nutrition_facts = ast.literal_eval(nutrition_facts_str)  # Nutrition Facts? ????? ??
                return description, nutrition_facts
    except Exception as e:
        logging.info(f"Error processing title: {title}, Error: {e}")
    return None, None  

def add_units_to_nutrition_facts(nutrition_facts):
    """Nutrition Facts? ??? ??"""
    if not isinstance(nutrition_facts, dict):
        return nutrition_facts  
    with_units = {}
    for key, value in nutrition_facts.items():
        unit = NUTRITION_UNITS.get(key, '')  
        with_units[key] = f"{value} {unit}" if unit else value
    return with_units

def extract_required_keys(nutrition_str):
    try:
        nutrition_dict = ast.literal_eval(nutrition_str)
        
        amount_per_serving = nutrition_dict.get('Amount per Serving', {})
        servings = nutrition_dict.get('Servings', 'Unknown') 
        extracted_data = {'Servings': servings} 

        for key in REQUIRED_KEYS:
            if key == 'Servings': 
                continue
            if key in ['Total Fat', 'Total Carbohydrates']:
                extracted_data[key] = amount_per_serving.get(key, {}).get('Amount', '0g')
                if key == 'Total Fat':
                    extracted_data['Saturated Fat'] = amount_per_serving.get(key, {}).get('Saturated Fat', '0g')
                    extracted_data['Trans Fat'] = amount_per_serving.get(key, {}).get('Trans Fat', '0g')
                if key == 'Total Carbohydrates':
                    extracted_data['Dietary Fiber'] = amount_per_serving.get(key, {}).get('Dietary Fiber', '0g')
                    extracted_data['Total Sugars'] = amount_per_serving.get(key, {}).get('Total Sugars', '0g')
                    extracted_data['Added Sugars'] = amount_per_serving.get(key, {}).get('Added Sugars', '0g')
            else:
                extracted_data[key] = amount_per_serving.get(key, '0')

        return extracted_data
    except Exception as e:
        logging.info(f"Error processing nutrition facts: {nutrition_str}, Error: {e}")
        return None
    
def normalize_numbers(nutrition_data):
    try:
        if isinstance(nutrition_data, str): 
            nutrition_data = ast.literal_eval(nutrition_data)
        if isinstance(nutrition_data, dict):
            for key, value in nutrition_data.items():
                if isinstance(value, dict):
                    nutrition_data[key] = normalize_numbers(value) 
                elif isinstance(value, str) and any(char.isdigit() for char in value): 
                    value = value.replace("g", "").replace("mg", "").replace("µg", "").replace("VALUE", "")
                    try:
                        nutrition_data[key] = float(value)
                    except ValueError:
                        nutrition_data[key] = value
                elif isinstance(value, (int, float)): 
                    nutrition_data[key] = float(value)
        return nutrition_data
    except Exception as e:
        logging.info(f"Error normalizing numbers: {nutrition_data}, Error: {e}")
        return {}

def parse_nutrition_facts(nutrition_data):
    try:
        if isinstance(nutrition_data, dict):  
            return nutrition_data
        elif isinstance(nutrition_data, str): 
            return ast.literal_eval(nutrition_data)
        else:  
            return {}
    except Exception as e:
        logging.info(f"Error parsing nutrition facts: {nutrition_data}, Error: {e}")
        return {}
    
def calculate_similarity(nutrition1, nutrition2):
    try:
        norm1 = normalize_numbers(nutrition1)
        norm2 = normalize_numbers(nutrition2)
        norm1_str = str(norm1)
        norm2_str = str(norm2)
        return SequenceMatcher(None, norm1_str, norm2_str).ratio()
    except Exception as e:
        logging.info(f"Error calculating similarity: {e}")
        return 0
    
def find_two_most_similar_titles(output_nutrition_facts, dfh):
    similarities = []
    for _, row in dfh.iterrows():
        nutrition_facts = parse_nutrition_facts(row['nutrition_facts'])
        similarity = calculate_similarity(output_nutrition_facts, nutrition_facts)
        similarities.append((similarity, row))
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:2]
    results = [(row['title'], row['description'], row['nutrition_facts'], sim) for sim, row in similarities]
    return results

def update_mm3_with_two_matches(mm, dfh):
    updated_rows = []
    for _, row in tqdm(mm.iterrows(), total=mm.shape[0]):
        processed_facts = parse_nutrition_facts(row['processed_output_nutrition_facts'])
        similar_matches = find_two_most_similar_titles(processed_facts, dfh)
        
        updated_row = row.copy()
        if len(similar_matches) > 0:
            updated_row['output_1'] = similar_matches[0][0]
            updated_row['output_description_1'] = similar_matches[0][1]
            updated_row['output_nutrition_facts_1'] = similar_matches[0][2]
        if len(similar_matches) > 1:
            updated_row['output_2'] = similar_matches[1][0]
            updated_row['output_description_2'] = similar_matches[1][1]
            updated_row['output_nutrition_facts_2'] = similar_matches[1][2]
        updated_rows.append(updated_row)
    return pd.DataFrame(updated_rows)

def extract_required_keys(nutrition_str):
    try:
        nutrition_dict = ast.literal_eval(nutrition_str)
        
        amount_per_serving = nutrition_dict.get('Amount per Serving', {})
        servings = nutrition_dict.get('Servings', 'Unknown')  
        extracted_data = {'Servings': servings} 

        for key in REQUIRED_KEYS:
            if key == 'Servings':  
                continue
            if key in ['Total Fat', 'Total Carbohydrates']:
                extracted_data[key] = amount_per_serving.get(key, {}).get('Amount', '0g')
                if key == 'Total Fat':
                    extracted_data['Saturated Fat'] = amount_per_serving.get(key, {}).get('Saturated Fat', '0g')
                    extracted_data['Trans Fat'] = amount_per_serving.get(key, {}).get('Trans Fat', '0g')
                if key == 'Total Carbohydrates':
                    extracted_data['Dietary Fiber'] = amount_per_serving.get(key, {}).get('Dietary Fiber', '0g')
                    extracted_data['Total Sugars'] = amount_per_serving.get(key, {}).get('Total Sugars', '0g')
                    extracted_data['Added Sugars'] = amount_per_serving.get(key, {}).get('Added Sugars', '0g')
            else:
                extracted_data[key] = amount_per_serving.get(key, '0')

        return extracted_data
    except Exception as e:
        print(f"Error processing nutrition facts: {nutrition_str}, Error: {e}")
        return None

def split_and_expand_rows(df):
    expanded_rows = []
    
    for _, row in df.iterrows():
        original_row = row.copy()
        original_row['row_type'] = 'original'
        expanded_rows.append(original_row.to_dict()) 

        new_row_1 = {
            'input': row['input'],
            'input_nutrition_facts': str(row['input_nutrition_facts']),
            'output': row['output_1'],
            'output_description': row['output_description_1'],
            'output_nutrition_facts': str(row['output_nutrition_facts_1']),
            'processed_output_nutrition_facts': str(row['processed_output_nutrition_facts_1']),
            'row_type': 'expanded_1'  # ???
        }
        expanded_rows.append(new_row_1)

        new_row_2 = {
            'input': row['input'],
            'input_nutrition_facts': str(row['input_nutrition_facts']),
            'output': row['output_2'],
            'output_description': row['output_description_2'],
            'output_nutrition_facts': str(row['output_nutrition_facts_2']),
            'processed_output_nutrition_facts': str(row['processed_output_nutrition_facts_2']),
            'row_type': 'expanded_2'  # ???
        }
        expanded_rows.append(new_row_2)

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df

def generate_reason(input_food, input_facts, output_food, output_desc, output_facts):
    try:
        prompt = (
            f"The consumed food '{input_food}' has certain nutritional deficiencies as detailed in: {input_facts}. "
            f"The recommended food '{output_food}' can help address these deficiencies based on the following details: "
            f"Description: {output_desc}, Nutrition Information: {output_facts}. "
            f"Explain briefly why '{output_food}' complements '{input_food}' without mentioning exact nutritional values. "
            f"If there are any notable drawbacks of '{output_food}', mention them as cautionary points. "
            f"Write a factual and concise explanation in 1-2 sentences."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def save_to_json(file_path, data):
    try:
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []  

        existing_data.append(data)

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"Error saving to JSON: {e}")

def process_and_save(file_path):
    for idx, row in tqdm(updated_mm3.iterrows(), total=updated_mm3.shape[0], desc="Processing rows"):
        reason = generate_reason(
            row['input'], row['input_nutrition_facts'],
            row['output'], row['output_description'],
            row['output_nutrition_facts']
        )

        result = {
            "input": row['input'],
            "input_nutrition_facts": row['input_nutrition_facts'],
            "output": row['output'],
            "output_description": row['output_description'],
            "output_nutrition_facts": row['output_nutrition_facts'],
            "reason": reason
        }

        save_to_json(file_path, result)

def generate_json_from_df(df):
    results = []
    for _, row in df.iterrows():
        result = {
            "instruction": "Based on the previous meal, suggest the next meal to maintain a balanced diet.",
            "input": row["input"],
            "output": f"{row['output']} is recommended. The reason is {row['reason']}"
        }
        results.append(result)
    return results

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # loading a snapme dataset
    mm = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data3_multimodal/version1/snapme_origin.csv")

    mm['food_identification_info'] = mm['food_identification_info'].astype(str) 

    mm = pd.DataFrame({
        'input': mm['food_identification_info'].apply(extract_food_description),
        'input_nutrition_facts': mm['nutritional_info'],
        'output': mm['recommended_meal']
    })

    NUTRITION_UNITS = {
        'PROT': 'g',   # ???
        'TFAT': 'g',   # ? ??
        'CARB': 'g',   # ????
        'SUGR': 'g',   # ?
        'FIBE': 'g',   # ????
        'MAGN': 'mg',  # ????
        'POTA': 'mg',  # ??
        'VB1': 'mg',   # ??? B1
        'VB6': 'mg',   # ??? B6
        'VB12': 'µg',  # ??? B12
        'SFAT': 'g'    # ?? ??
    }

    output_descriptions = []
    output_nutrition_facts_with_units = []

    for i, row in mm.iterrows():
        title = row['recommended_meal']
        context = row['context_info']
        description, nutrition_facts = extract_context_info(context, title)
        output_descriptions.append(description)
        if nutrition_facts:
            try:
                nutrition_with_units = add_units_to_nutrition_facts(nutrition_facts)
                output_nutrition_facts_with_units.append(nutrition_with_units)
            except Exception as e:
                print(f"Error adding units to nutrition facts for {title}, Error: {e}")
                output_nutrition_facts_with_units.append(None)
        else:
            output_nutrition_facts_with_units.append(None)

    mm['input_nutrition_facts'] = mm['input_nutrition_facts'].apply(lambda x: add_units_to_nutrition_facts(ast.literal_eval(x)))

    mm['output_description'] = output_descriptions
    mm['output_nutrition_facts'] = output_nutrition_facts_with_units

    # loading a dfh dataset
    dfh = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data2_daily_diets/diabetes_food_hub_new_nutri_facts.csv")
    dfh = dfh.drop_duplicates(subset='title')
    dfh_tmp = dfh[['title', 'nutrition_facts']].drop_duplicates()
    mm = pd.merge(mm, dfh_tmp, left_on='output', right_on='title', how='left')

    mm['nutrition_facts'].isna().sum()
    mm = mm[['input', 'input_nutrition_facts', 'output', 'output_description', 'nutrition_facts']]
    mm = mm.rename(columns={'nutrition_facts': 'output_nutrition_facts'})
    mm = mm.dropna(subset=['output_nutrition_facts'])

    REQUIRED_KEYS = [
        'Calories',              # ???
        'Total Fat',             # ? ??
        'Saturated Fat',         # ?? ??
        'Trans Fat',             # ??? ??
        'Cholesterol',           # ?????
        'Sodium',                # ???
        'Total Carbohydrates',   # ? ????
        'Dietary Fiber',         # ????
        'Total Sugars',          # ??
        'Added Sugars',          # ???
        'Protein',               # ???
        'Potassium',             # ??
        'Servings'               # ?? ?? ??
    ]

    mm['processed_output_nutrition_facts'] = mm['output_nutrition_facts'].apply(extract_required_keys)

    mm = update_mm3_with_two_matches(mm, dfh) # It takes about 40 minutes
    
    # mm.to_csv("/data/jaesung/llm_for_diabetes/src/data/data3_multimodal/version1/tmp_updated_mm3.csv")

    # loading a mm dataset
    updated_mm3 = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data3_multimodal/version1/tmp_updated_mm3.csv")

    REQUIRED_KEYS = [
        'Calories',              # ???
        'Total Fat',             # ? ??
        'Saturated Fat',         # ?? ??
        'Trans Fat',             # ??? ??
        'Cholesterol',           # ?????
        'Sodium',                # ???
        'Total Carbohydrates',   # ? ????
        'Dietary Fiber',         # ????
        'Total Sugars',          # ??
        'Added Sugars',          # ???
        'Protein',               # ???
        'Potassium',             # ??
        'Servings'               # ?? ?? ??
    ]

    mm['processed_output_nutrition_facts'] = mm['output_nutrition_facts'].apply(extract_required_keys)
    mm['processed_output_nutrition_facts_1'] = mm['output_nutrition_facts_1'].apply(extract_required_keys)
    mm['processed_output_nutrition_facts_2'] = mm['output_nutrition_facts_2'].apply(extract_required_keys)

    mm = mm[['input', 'input_nutrition_facts', 'output',
        'output_description', 'output_nutrition_facts',
        'processed_output_nutrition_facts', 'output_1', 'output_description_1',
        'output_nutrition_facts_1', 'processed_output_nutrition_facts_1', 'output_2', 'output_description_2',
        'output_nutrition_facts_2', 'processed_output_nutrition_facts_2']]
    
    mm = split_and_expand_rows(mm)

    mm = mm[['input', 'input_nutrition_facts', 'output', 'output_description', 'output_nutrition_facts', 'processed_output_nutrition_facts']]

    mm = mm[updated_mm3['output_description']!='Description not found']

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    file_path = "snapme_origin.json"
    process_and_save(file_path)

    # loading a dataset preprocessed data
    file_path = "/data/jaesung/llm_for_diabetes/src/data/data3_multimodal/version1/snapme_origin.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        temp = json.load(f)

    mm = pd.DataFrame(temp)

    input_groups = mm.groupby('input')
    unique_inputs = list(input_groups.groups.keys())
    train_inputs, test_inputs = train_test_split(unique_inputs, test_size=0.2, random_state=42)

    train_df = mm[mm['input'].isin(train_inputs)].copy()
    test_df = mm[mm['input'].isin(test_inputs)].copy()

    json_results = generate_json_from_df(train_df)
    json_results = generate_json_from_df(test_df)

    # save the datasets
    with open("snapme_train_instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    with open("snapme_test_instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)