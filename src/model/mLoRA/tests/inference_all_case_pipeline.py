import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

G_TEST_ADAPTERS = [
    # lora adapter
    "/data/jaesung/llm_for_diabetes/src/model/mLoRA/adapters/lora_sft_0",
    "/data/jaesung/llm_for_diabetes/src/model/mLoRA/adapters/lora_sft_1",
]


def get_cmd_args():
    parser = argparse.ArgumentParser(description='mLoRA test function')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to or name of base model')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()

    model_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)


    # load adapter
    for adapter in G_TEST_ADAPTERS:
        model.load_adapter(adapter, adapter_name=adapter)
    model.disable_adapters()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        # device_map="auto",
        device=0
    )

    model.enable_adapters()

    for adapter in G_TEST_ADAPTERS:
        model.set_adapter(adapter)

        # ???? 2?? ???? "input" ?? ??? ??
        input_text = (
        "Instruction: As a medical AI assistant, generate a professional and medically accurate response to the given patient query. Ensure that your response aligns with medical guidelines and provides clear, practical advice. Avoid assumptions and base your answer solely on the provided information."
        "Input: Hello doctor, I have diabetes since the last 11 to 12 years, and I started taking medicines given by my family doctor. Presently, I am taking Glimisave M2 (one tablet in the morning and evening), Dynaglipt 20 (one tablet in the morning), and Absolut 3G (one in the evening). My health is fine, and I do not have any other diseases. Recently, my blood sugar level showed up high, so my family doctor advised me to consult a diabetologist to control blood sugar level. Please help me. I have attached my test reports."
        "Output: "
        )

        sequences = pipeline(
            input_text,
            do_sample=True,
            top_k=10, # 50
            num_return_sequences=4,
            temperature=0.7, # 0.7
            top_p=0.9, # 0.9
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024, # 256
            truncation=True,
            # repetition_penalty=1.2,
        )

        # print(sequences)

        # 'generated_text'?? "answer: " ?? ?? ??
        output = sequences[0]['generated_text'].strip()
        if "Output:" in output:
            output = output.split("Output:")[-1].strip()

        print(f"Adapter {adapter} Output is: {output}")
