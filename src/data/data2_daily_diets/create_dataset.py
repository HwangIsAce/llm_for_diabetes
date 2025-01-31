import ast
import json
import logging
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def create_instruction_dataset_from_row(row):
    instructions = []

    try:
        nutrition_facts = {
            "Amount per serving": row["nutrition_facts.Amount per serving"],
            "Total Fat": row["nutrition_facts.Total Fat"],
            "Cholesterol": row["nutrition_facts.Cholesterol"],
            "Sodium": row["nutrition_facts.Sodium"],
            "Total Carbohydrate": row["nutrition_facts.Total Carbohydrate"],
            "Protein": row["nutrition_facts.Protein"],
        }
        nutrition_output = ", ".join([f"{key}: {value}" for key, value in nutrition_facts.items()])
    except Exception as e:
        logging.info(f"Error parsing nutrition facts: {e}")
        nutrition_output = "N/A"

    instructions.append({
        "instruction": "Provide the nutrition facts for the given recipe.",
        "input": row["title"],
        "output": nutrition_output,
    })

    try:
        ingredients = ast.literal_eval(row["ingredients"])
        if isinstance(ingredients, list) and all(isinstance(ing, dict) for ing in ingredients):
            ingredients_output = "\n".join(
                [f"- {ing['label']}: {ing.get('us_measure', 'N/A')} ({ing.get('metric_measure', 'N/A')})" for ing in ingredients]
            )
        else:
            ingredients_output = "N/A"
    except Exception as e:
        logging.info(f"Error parsing ingredients: {row['ingredients']} - {e}")
        ingredients_output = "N/A"

    instructions.append({
        "instruction": "List the measurements for all ingredients in the given recipe.",
        "input": row["title"],
        "output": ingredients_output,
    })

    try:
        tags = ast.literal_eval(row["tags"])
        if isinstance(tags, list):
            tags_output = ", ".join(tags)
        else:
            tags_output = "N/A"
    except Exception as e:
        logging.info(f"Error parsing tags: {row['tags']} - {e}")
        tags_output = "N/A"

    instructions.append({
        "instruction": "What tags are associated with this recipe?",
        "input": row["title"],
        "output": tags_output,
    })

    try:
        description_output = row["description"] if isinstance(row["description"], str) else "N/A"
    except Exception as e:
        logging.info(f"Error parsing description: {e}")
        description_output = "N/A"

    instructions.append({
        "instruction": "Provide a brief description of the given recipe.",
        "input": row["title"],
        "output": description_output,
    })

    return instructions

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_line_by_line(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def mmr_selection(documents, n_samples, lambda_param=0.5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    selected_indices = []
    remaining_indices = list(range(len(documents)))

    selected_indices.append(remaining_indices.pop(0))

    for _ in tqdm(range(n_samples - 1), desc="Selecting samples"):
        max_mmr_score = -float('inf')
        best_idx = -1
        
        for idx in remaining_indices:
            relevance = np.mean([similarity_matrix[idx][i] for i in selected_indices])
            diversity = np.min([similarity_matrix[idx][i] for i in selected_indices])
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices

def process_json(input_file, output_file, n_samples=5000):
    data = load_json(input_file)

    documents = []
    for day_plan in data:
        documents.append(json.dumps(day_plan, ensure_ascii=False))

    logging.info("Calculating MMR...")
    start_time = time.time()
    selected_indices = mmr_selection(documents, n_samples)
    end_time = time.time()

    elapsed_time = end_time - start_time
    logging.info(f"MMR calculation completed in {elapsed_time:.2f} seconds.")

    sampled_data = [data[i] for i in selected_indices]

    logging.info("Saving selected samples...")
    save_json_line_by_line(sampled_data, output_file)
    logging.info(f"Saved {n_samples} samples to {output_file}")

def calculate_daily_total(diet):
    daily_total = {
        "Calories": 0,
        "Carbohydrate (g)": 0,
        "Fiber (g)": 0,
        "Protein (g)": 0,
        "Fat (g)": 0,
        "Cholesterol (mg)": 0,
        "Sodium (mg)": 0,
        "Potassium (mg)": 0
    }
    
    for meal_type in ["Breakfast", "Lunch", "Dinner"]:
        if meal_type in diet and "Nutrition" in diet[meal_type]:
            nutrition = diet[meal_type]["Nutrition"]
            
            # Calories
            calories = nutrition.get("Calories", "0").replace("g", "").replace("mg", "")
            daily_total["Calories"] += float(calories) if calories.isdigit() else 0
            
            # Carbohydrate (g)
            carbohydrate = nutrition.get("Total Carbohydrate", {}).get("Amount", "0g").replace("g", "")
            daily_total["Carbohydrate (g)"] += float(carbohydrate) if carbohydrate.isdigit() else 0

            # Fiber 
            fiber = nutrition.get("Total Carbohydrate", {}).get("Dietary Fiber", "0g").replace("g", "")
            daily_total["Fiber (g)"] += float(fiber) if fiber.isdigit() else 0
            
            # Protein (g)
            protein = nutrition.get("Protein", "0g").replace("g", "")
            daily_total["Protein (g)"] += float(protein) if protein.isdigit() else 0
            
            # Fat (g)
            fat = nutrition.get("Total Fat", {}).get("Amount", "0g").replace("g", "")
            daily_total["Fat (g)"] += float(fat) if fat.isdigit() else 0
            
            # Cholesterol (mg)
            cholesterol = nutrition.get("Cholesterol", "0mg").replace("mg", "")
            daily_total["Cholesterol (mg)"] += float(cholesterol) if cholesterol.isdigit() else 0
            
            # Sodium (mg)
            sodium = nutrition.get("Sodium", "0mg").replace("mg", "")
            daily_total["Sodium (mg)"] += float(sodium) if sodium.isdigit() else 0
            
            # Potassium (mg)
            potassium = nutrition.get("Potassium", "0mg")
            if potassium is None:
                potassium = "0mg"
            potassium = potassium.replace("mg", "")
            daily_total["Potassium (mg)"] += float(potassium) if potassium.isdigit() else 0
    
    return daily_total     

def parse_nutrition_facts(row):
    try:
        if isinstance(row["nutrition_facts"], str):
            nutrition_facts = ast.literal_eval(row["nutrition_facts"])
        elif isinstance(row["nutrition_facts"], dict):
            nutrition_facts = row["nutrition_facts"]
        else:
            nutrition_facts = {}

        amount_per_serving = nutrition_facts.get("Amount per Serving", {})
        nutrition_facts_parsed = {
            "Calories": amount_per_serving.get("Calories", "N/A"),
            "Total Fat": amount_per_serving.get("Total Fat", {}),
            "Cholesterol": amount_per_serving.get("Cholesterol", "N/A"),
            "Sodium": amount_per_serving.get("Sodium", "N/A"),
            "Total Carbohydrate": amount_per_serving.get("Total Carbohydrates", {}),
            "Protein": amount_per_serving.get("Protein", "N/A"),
            "Potassium": amount_per_serving.get("Potassium", "N/A"),
        }
        return nutrition_facts_parsed
    except Exception as e:
        logging.info(f"Error parsing nutrition facts: {e}")
        return {}

def update_json_from_merged_df(df, daily_diets):
    for diet in daily_diets:
        for meal_type in ["Breakfast", "Lunch", "Dinner"]:
            if meal_type in diet:
                dish_name = diet[meal_type]["Dish"]

                matching_row = df[df["title"] == dish_name]
                if not matching_row.empty:
                    row = matching_row.iloc[0]
                    nutrition_facts = parse_nutrition_facts(row)
                    
                    diet[meal_type]["Nutrition"] = nutrition_facts
        
        diet["Daily Total"] = calculate_daily_total(diet)
    
    return daily_diets

def filter_diets_by_macros(daily_diets, carb_range, protein_range, fat_range):

    filtered_diets = []

    for diet in daily_diets:
        daily_total = diet.get('Daily Total', {})

        carbs = daily_total.get('Carbohydrate (g)', 0)
        protein = daily_total.get('Protein (g)', 0)
        fat = daily_total.get('Fat (g)', 0)
        calories = daily_total.get('Calories', 1)  # Avoid division by zero

        carb_ratio = (carbs * 4 / calories) * 100
        protein_ratio = (protein * 4 / calories) * 100
        fat_ratio = (fat * 9 / calories) * 100

        if (
            carb_range[0] <= carb_ratio <= carb_range[1]
            and protein_range[0] <= protein_ratio <= protein_range[1]
            and fat_range[0] <= fat_ratio <= fat_range[1]
        ):
            filtered_diets.append(diet)

    return filtered_diets

def custom_tokenizer(text):
    return text.split("\n")

def get_main_ingredients(ingredients_list):
    all_ingredients = [ingredient['label'] for meal in ingredients_list for ingredient in meal]
    all_ingredients_combined = "\n".join(all_ingredients)

    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)  
    tfidf_matrix = vectorizer.fit_transform([all_ingredients_combined])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    ingredient_scores = dict(zip(feature_names, scores))

    sorted_ingredients = sorted(ingredient_scores, key=ingredient_scores.get, reverse=True)
    return sorted_ingredients[:10]

def drop_duplicate_and_sample_instruction(dataset):
    instruction_dataset = []
    for idx, day in enumerate(dataset):
        breakfast = day["Breakfast"]
        lunch = day["Lunch"]
        dinner = day["Dinner"]
        daily_nutrition = day["Daily Total"]

        ingredients_list = [
            breakfast["Ingredients"],
            lunch["Ingredients"],
            dinner["Ingredients"]
        ]

        main_ingredients = get_main_ingredients(ingredients_list)

        input_goals = (
            f"Ensure the daily carbohydrate intake does not exceed {daily_nutrition['Carbohydrate (g)']}g, "
            f"protein intake is at least {daily_nutrition['Protein (g)']}g, and fat intake does not exceed {daily_nutrition['Fat (g)']}g."
        )

        instruction_dataset.append({
            "instruction": "Recommend a daily diet based on the given nutritional goals.",
            "input": input_goals,
            "output": {
                "Breakfast": breakfast["Dish"],
                "Lunch": lunch["Dish"],
                "Dinner": dinner["Dish"]
            }
        })

        selected_ingredients = random.sample(main_ingredients, min(3, len(main_ingredients)))

        for ingredient in selected_ingredients:
            clean_ingredient = ingredient.replace("\\", "").replace("\"", "") 
            instruction_dataset.append({
                "instruction": "Recommend a daily diet that includes a specific ingredient.",
                "input": f"Create a diet that includes {clean_ingredient}.",
                "output": {
                    "Breakfast": breakfast["Dish"],
                    "Lunch": lunch["Dish"],
                    "Dinner": dinner["Dish"]
                }
            })

        instruction_dataset.append({
            "instruction": "Analyze and summarize the nutritional content of a daily diet.",
            "input": {
                "Breakfast": breakfast["Dish"],
                "Lunch": lunch["Dish"],
                "Dinner": dinner["Dish"]
            },
            "output": {
                "Daily Total": daily_nutrition
            }
        })

        return instruction_dataset


if __name__ == "__main__":

    # logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # loading datasets
    dfh = pd.read_csv("/data/jaesung/llm_for_diabetes/src/data/data2_daily_diets/diabetes_food_hub_nutri_facts.csv")
    dfh.drop_duplicates(['title'], inplace=True)

    dfh_instruction_dataset = []

    for idx, row in dfh.iterrows():
        dfh_instruction_dataset.extend(create_instruction_dataset_from_row(row))

    dfh_instruction_dataset = pd.DataFrame(dfh_instruction_dataset)

    # filtering using MMR search (It takes a lot of time)
    input_file = "/data/jaesung/llm_for_diabetes/src/data/data2_daily_diets/daily_diets_drop_duplicated.json"
    output_file = "daily_diets_drop_duplicated_and_sampled.json"
    
    process_json(input_file, output_file)

    # loading datasets filtered by MMR search
    file_path = "/data/jaesung/llm_for_diabetes/src/data/data2_daily_diets/daily_diets_drop_duplicated_and_sampled.json"
    daily_diets = []
    with open(file_path , 'r', encoding='utf-8') as f:
        for line in f:
            daily_diets.append(json.loads(line.strip()))


    with open(file_path, 'r', encoding='utf-8') as f:
        daily_diets = [json.loads(line.strip()) for line in f]

    updated_daily_diets = update_json_from_merged_df(dfh, daily_diets)

    output_path = "daily_diets_drop_duplicated_and_sampled_and_modified_nutri_facts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        for diet in updated_daily_diets:
            f.write(json.dumps(diet, ensure_ascii=False) + "\n")

    # loading datasets filtered by MMR search and modified by nutrition facts
    file_path = "/data/jaesung/llm_for_diabetes/src/data/data2_daily_diets/daily_diets_drop_duplicated_and_sampled_and_modified_nutri_facts.json"
    daily_diets = []
    with open(file_path , 'r', encoding='utf-8') as f:
        for line in f:
            daily_diets.append(json.loads(line.strip()))

    carb_range = (35, 65)  # Carbohydrate ratio: 50%
    protein_range = (15, 45)  # Protein ratio: 30%
    fat_range = (5, 45)  # Fat ratio: 20%

    filtered_diets = filter_diets_by_macros(daily_diets, carb_range, protein_range, fat_range)

    # loading datasets filtered by MMR search and modified by nutrition facts and filtered by macro
    with open('daily_diets_drop_duplicated_and_sampled_and_modified_nutri_facts_and_sampled.json', 'w') as f:
        daily_diets = json.dump(filtered_diets, f, indent=4)

    daily_diets_train, daily_diets_test = train_test_split(daily_diets, test_size=0.2, random_state=42)

    daily_diets_train = drop_duplicate_and_sample_instruction(daily_diets_train)
    daily_diets_test = drop_duplicate_and_sample_instruction(daily_diets_test)

    # save the datasets
    with open("daily_diets_drop_duplicated_and_sampled_train_instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(daily_diets_train, f, ensure_ascii=False, indent=4)
    with open("daily_diets_drop_duplicated_and_sampled_test_instruction_dataset.json", "w", encoding="utf-8") as f:
        json.dump(daily_diets_test, f, ensure_ascii=False, indent=4)

    