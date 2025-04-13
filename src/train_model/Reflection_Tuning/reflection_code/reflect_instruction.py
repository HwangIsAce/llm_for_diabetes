import argparse
import json
import os
import time
import openai
from tqdm import tqdm
import asyncio
import logging
from typing import List, Dict, Any

import tiktoken
from transformers import LlamaTokenizer

# GPT Tokenizer ??
gpt_encoder = tiktoken.get_encoding("cl100k_base")
# llama_tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
llama_tokenizer = LlamaTokenizer.from_pretrained('baffo32/decapoda-research-llama-7B-hf')

# ?? ??
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """OpenAI API ??? ??"""
    async_responses = []
    timeout_seconds = 60

    for x in messages_list:
        try:
            response = await asyncio.wait_for(
                openai.ChatCompletion.acreate(
                    model=model,
                    messages=x,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ),
                timeout=timeout_seconds
            )
            async_responses.append(response)

        except asyncio.TimeoutError:
            logger.error("OpenAI API ?? ?? ?? (Timeout)")
            async_responses.append({"choices": [{"message": {"content": "[Error] API ?? ?? ??"}}]})

        except Exception as e:
            logger.error(f"OpenAI API ?? ??: {e}")
            async_responses.append({"choices": [{"message": {"content": "[Error] API ?? ??"}}]})

    return async_responses

def gen_prompt_no_input(ins, outp):
    """???? ??"""
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = (
        "We would like you to answer several questions related to the quality of a given instruction. \n"
        "1. Why this instruction is not good? First analyse the instruction based on Complexity of the Topic, Level of Detail Required, Knowledge Required, Ambiguity of the Instruction and Logical Reasoning or Problem-Solving Involved.\n"
        "Then analyse why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details.\n"
        "Finally analyse why this bad instruction lead to a bad answer.\n"
        "2. Based on the reason you provided, generate a new and complete instruction which is complex and difficult to answer directly.\n"
        "Make sure the new instruction is relevant but independent to the original instruction, put the new instruction in the format of [New Instruction] your instruction [End]\n"
        "3. Answer the newly generated instruction as detailed as possible, in the format of [New Answer] your answer [End]\n"
    )
    return sys_prompt, prompt_template.format(ins=ins, outp=outp, criteria=criteria)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--api_base", type=str, default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)

    args = parser.parse_args()
    if args.api_base:
        openai.api_base = args.api_base
    openai.api_key = args.api_key

    # ??? ??
    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    # JSONL ?? ??
    jsonl_save_path = args.save_path.replace(".json", ".jsonl")

    message_list = []
    token_len_list = []
    
    for i, data_i in enumerate(sampled_data):
        instruct_i = data_i['instruction'].strip()
        output_i_raw = data_i['output']
        if isinstance(output_i_raw, list):
            output_i = " ".join(output_i_raw).strip()
        else:
            output_i = output_i_raw.strip()
        input_i = data_i.get('input', '').strip()

        whole_text = instruct_i + input_i + output_i
        inputs = llama_tokenizer(whole_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > 2048:
            gap = input_ids.shape[1] - 2048
            output_i = output_i[:-gap]

        sys_prompt, prompt = gen_prompt_no_input(instruct_i, output_i)
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        message_list.append(message)
        token_len_list.append(len(gpt_encoder.encode(prompt)))

    predictions = []
    i = 0
    wait_base = 10
    max_wait = 60  # ?? ?? ?? ??
    retry = 0
    error = 0
    pbar = tqdm(total=len(message_list))
    batch_size = args.batch_size

    # JSONL ??? ???? ?? ??? ??
    processed_count = 0
    if os.path.exists(jsonl_save_path):
        with open(jsonl_save_path, "r") as f:
            processed_count = sum(1 for _ in f)
        logger.info(f"?? ??? ???: {processed_count}?")

    while i < len(message_list):
        if i < processed_count:  # ?? ??? ?? ??
            i += batch_size
            pbar.update(batch_size)
            continue

        token_limit_in_current_batch = min(args.max_tokens, 4050 - max(token_len_list[i:i + batch_size]))
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i + batch_size],
                    model=args.api_model,
                    temperature=0.0,
                    max_tokens=token_limit_in_current_batch,
                    top_p=1.0,
                )
            )
            for prediction in batch_predictions:
                review = prediction['choices'][0]['message']['content']
                with open(jsonl_save_path, "a") as f:
                    f.write(json.dumps({"response": review}) + "\n")  # JSONL ???? ??
            predictions += batch_predictions
            retry = 0
            i += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except Exception as e:
            retry += 1
            error += 1
            logger.error(f"Batch error at index {i}-{i + batch_size}: {e}")
            logger.error(f"Retry number: {retry}, Error count: {error}")
            time.sleep(wait_base)
            wait_base = min(wait_base * 2, max_wait)  # ?? ???? ??

    pbar.close()
    print(f"? {len(predictions)} ?? ???? ???????.")
