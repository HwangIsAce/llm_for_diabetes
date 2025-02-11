import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# ?? ??
if __name__ == "__main__":
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

    # ?? ???? ??
    input_text = (
        "instruction: Generate a structured daily meal plan for diabetes management. The recommended meals must be selected only from the provided dataset and should not include any additional information.\n\nThe output must strictly follow this format:\n\nBreakfast: [meal]\nLunch: [meal]\nDinner: [meal]\n\nThen, provide a brief explanation of how the selected meals support blood sugar control. The explanation should be concise and relevant to diabetes management, focusing on how the selected meals balance macronutrients."
        "input: Create a meal plan that includes chicken tenderloins((about 1 pound)), maintaining a 50:30:20 macronutrient balance and supporting blood sugar stability for individuals with diabetes."
        "output:"
    )

    # ??? ??? ??? ? ??? ??
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")  # ? ??

    # # **model.generate() ?? (pipeline() ??)**
    # output_ids = model.generate(
    #     input_ids,
    #     attention_mask=attention_mask,  # ? ???? ?? ??? ???? ????? ?
    #     max_new_tokens=1024, 
    #     do_sample=True,  # ? ?? ??? ???
    #     num_return_sequences=2,  # ? ? ?? ??? ????? ??
    #     temperature=0.7, 
    #     top_k=40,
    #     top_p=0.95,
    #     repetition_penalty=1.1,
    #     eos_token_id=tokenizer.eos_token_id
    # )

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,  # ? ??? ?? ??
        num_return_sequences=2,  # ? ? ?? ??? ??? ??
        do_sample=True,  # ? ??? ??? ???
        temperature=0.5,  # ? ?? ?? ???? ??
        top_k=50,
        top_p=0.5,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id  # ? ?? ?? ??
    )

    # ??? ??? ??? & Output ??? ??
    output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

    print(f"\n=== Adapter {G_TEST_ADAPTERS} Outputs ===\n")
    for i, output_text in enumerate(output_texts, 1):
        # ? "Output:" ??? ??? ??
        if "output:" in output_text:
            output_text = output_text.split("output:")[-1].strip()
        
        print(f"?? **Generated Response {i}:**\n{output_text}\n")
        print("=" * 80)
