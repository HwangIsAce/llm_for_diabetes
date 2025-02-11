import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import time

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
# def save_json(data, file_path):
#     with open(file_path, "w", encoding="utf-8") as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)

def save_to_jsonl(file_path, data):
    with open(file_path, "a", encoding="utf-8") as f: 
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# LoRA ??? ??
G_TEST_ADAPTERS = [
    "/data/jaesung/llm_for_diabetes/src/model/mLoRA/adapters/lora_sft_0",
    "/data/jaesung/llm_for_diabetes/src/model/mLoRA/adapters/lora_sft_1"
]

# ??? ?? ??
def get_cmd_args():
    parser = argparse.ArgumentParser(description="mLoRA test function")
    parser.add_argument("--base_model", type=str, required=True, help="Path to or name of base model")
    return parser.parse_args()

def generate_response(task, instruction, input_text, max_tokens):
    full_input = f"instruction: {instruction}\ninput: {input_text}\noutput: "  

    inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    output_prefix =""
    if task in ["qa1", "qa2"]:
        output_prefix = "A or B or C or D"
    elif task == "qa3":
        output_prefix = "Yes or No"
    elif task == "nli":
        output_prefix = "Entailment or Contradiction"
    elif task == "re":
        output_prefix = "DDI-false or DDI-effect or DDI-advice or DDI-int or DDI-mechanism)"

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,  # ? ??? ?? ??
        num_return_sequences=2,  # ? ? ?? ??? ??? ??
        do_sample=True,  # ? ??? ??? ???
        temperature=0.9,  # ? ?? ?? ???? ??
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # ? ?? ?? ??

    )
    torch.cuda.empty_cache()  

    responses = [tokenizer.decode(output_ids[i], skip_special_tokens=True) for i in range(2)]

    processed_responses = []
    for response in responses:
        if "output:" in response.lower():  # ???? ?? ?? "output:" ??? ??
            response = response.lower().split("output:")[-1].strip()

        # Prefix? ?? ?? ?? (None ?? ??)
        # if output_prefix and not response.startswith(output_prefix.strip()):
        #     response = output_prefix + response

        if output_prefix and response.lower().startswith(output_prefix.lower()):
            response = response[len(output_prefix):].strip()

        processed_responses.append(response.strip())

    return processed_responses

# ?? ??
if __name__ == "__main__":

    input_json_path = "/data/jaesung/llm_for_diabetes/src/model/mLoRA/demo/data_classification_test.json"
    output_jsonl_path = "/data/jaesung/llm_for_diabetes/src/model/mLoRA/model_output/classification_model_output.jsonl"

    data = load_json(input_json_path)

    args = get_cmd_args()
    model_path = args.base_model

    # ?? & ????? ??
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")  # GPU ??

    # LoRA ??? ??
    adapter_names = []
    for adapter in G_TEST_ADAPTERS:
        model.load_adapter(adapter, adapter_name=adapter)
        adapter_names.append(adapter)

    # ? ?? ?? ???? ??? ???
    model.set_adapter(adapter_names)

    start_time = time.time()
    total_samples = len(data)
    for idx, item in enumerate(data):
        sample_start_time = time.time()

        input_text = item.get("input", "")  
        instruction = item.get("instruction", "") 
        task = item.get("task", "")
    
        res_4 = generate_response(task, instruction, input_text, 4)
        res_128 = generate_response(task, instruction, input_text, 128)
        res_2048 = generate_response(task, instruction, input_text, 2048)

        output_data = item.copy()
        output_data.update({
            "model_res1_4": res_4[0],
            "model_res2_4": res_4[1],
            "model_res1_128": res_128[0],
            "model_res2_128": res_128[1],
            "model_res1_2048": res_2048[0],
            "model_res2_2048": res_2048[1],
        })
        save_to_jsonl(output_jsonl_path, output_data)


        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / (idx + 1) 
        remaining_samples = total_samples - (idx + 1)
        estimated_remaining_time = remaining_samples * avg_time_per_sample

        print(f"[{idx+1}/{total_samples}] Sample processed in {time.time() - sample_start_time:.2f}s, ETA: {estimated_remaining_time/60:.2f} min")

    print(f"\nAll samples processed. Total time: {(time.time() - start_time)/60:.2f} min")
