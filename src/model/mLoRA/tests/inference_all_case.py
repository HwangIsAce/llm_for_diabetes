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
        "instruction: Respond to the patient's query as a professional medical AI, ensuring accuracy and adherence to medical standards. Offer clear, evidence-based guidance while avoiding speculation. Your response should rely only on the provided information and align with established medical protocols.\n"
        "input: Hello doctor, I have diabetes since the last 11 to 12 years, and I started taking medicines given by my family doctor. Presently, I am taking Glimisave M2 (one tablet in the morning and evening), Dynaglipt 20 (one tablet in the morning), and Absolut 3G (one in the evening). My health is fine, and I do not have any other diseases. Recently, my blood sugar level showed up high, so my family doctor advised me to consult a diabetologist to control blood sugar level. Please help me. I have attached my test reports.\n"
        "output:"
    )

    # ??? ??? ??? ? ??? ??
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")  # ? ??

    # **model.generate() ?? (pipeline() ??)**
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,  # ? ???? ?? ??? ???? ????? ?
        max_new_tokens=64, 
        do_sample=True,  # ? ?? ??? ???
        num_return_sequences=2,  # ? ? ?? ??? ????? ??
        temperature=0.7, 
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    # output_ids = model.generate(
    #     input_ids,
    #     attention_mask=attention_mask,
    #     max_new_tokens=2048,  # ? ??? ?? ??
    #     num_return_sequences=2,  # ? ? ?? ??? ??? ??
    #     do_sample=True,  # ? ??? ??? ???
    #     temperature=0.7,  # ? ?? ?? ???? ??
    #     top_k=40,
    #     top_p=0.9,
    #     repetition_penalty=1.1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.eos_token_id  # ? ?? ?? ??
    # )

    # ??? ??? ??? & Output ??? ??
    output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

    print(f"\n=== Adapter {G_TEST_ADAPTERS} Outputs ===\n")
    for i, output_text in enumerate(output_texts, 1):
        # ? "Output:" ??? ??? ??
        if "output:" in output_text:
            output_text = output_text.split("output:")[-1].strip()
        
        print(f"?? **Generated Response {i}:**\n{output_text}\n")
        print("=" * 80)
