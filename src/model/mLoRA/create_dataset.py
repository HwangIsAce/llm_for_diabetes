import os
import json
from dotenv import load_dotenv  # Import "dotenv" could not be resolved
from datasets import load_dataset  # Import "datasets" could not be resolved

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # load dataset
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")

    diabetes = load_dataset("passionMan/diabetes_v18", token=huggingface_token)

    # split dataset into classification group, generation group
    group_classification = {"qa1", "qa2", "qa3", "nli", "ie", "re"}
    group_generation = {"summarization", "generation", "daily_diets", "alternative_diet"}

    split_data = {"group1": {"train": [], "test": []}, "group2": {"train": [], "test": []}}

    for split in ["train", "test"]:
        for example in diabetes[split]:
            key = "group1" if example["task"] in group_classification else "group2"
            split_data[key][split].append(example)

    # data classification
    save_json(split_data["group1"]["train"], "demo/data_classification_train.json")
    save_json(split_data["group1"]["test"], "demo/data_classification_test.json")

    # data generation
    save_json(split_data["group2"]["train"], "demo/data_generation_train.json")
    save_json(split_data["group2"]["test"], "demo/data_generation_test.json")
