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
        "instruction: Extract the key insights from the given PubMed article and organize them into the following structured sections: Objective, Methods, Results, and Conclusion."
        "input: The American Diabetes Association and the European Association for the Study of Diabetes convened a panel to update the previous consensus statements on the management of hyperglycemia in type 2 diabetes in adults, published since 2006 and last updated in 2019. The target audience is the full spectrum of the professional health care team providing diabetes care in the U.S. and Europe. A systematic examination of publications since 2018 informed new recommendations. These include additional focus on social determinants of health, the health care system, and physical activity behaviors, including sleep. There is a greater emphasis on weight management as part of the holistic approach to diabetes management. The results of cardiovascular and kidney outcomes trials involving sodium-glucose cotransporter 2 inhibitors and glucagon-like peptide 1 receptor agonists, including assessment of subgroups, inform broader recommendations for cardiorenal protection in people with diabetes at high risk of cardiorenal disease. After a summary listing of consensus recommendations, practical tips for implementation are provided."
        "answer: "
        )

        sequences = pipeline(
            input_text,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            truncation=True,
        )

        # print(sequences)

        # 'generated_text'?? "answer: " ?? ?? ??
        output = sequences[0]['generated_text'].strip()
        if "answer:" in output:
            output = output.split("answer:")[-1].strip()

        print(f"Adapter {adapter} Output is: {output}")
