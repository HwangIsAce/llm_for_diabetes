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

# OpenAI ?? ??? & Llama Tokenizer ???
gpt_encoder = tiktoken.get_encoding("cl100k_base")
llama_tokenizer = LlamaTokenizer.from_pretrained('baffo32/decapoda-research-llama-7B-hf')

# ?? ??
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI ??? ?? (???? ??)
async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[Dict[str, Any]]:
    """OpenAI API? ??? ????, ?? 60? ????? ????."""
    
    async_responses = []
    timeout_seconds = 60  # ?? ?? ?? 60?

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

def gen_instruction_from_output(ins, outp):
    sys_prompt = (
        "You are a helpful assistant for improving the clarity and generalizability of instructions "
        "while preserving the original intent and aligning with the output."
    )

    prompt_template = (
        "[Original Instruction]\n{ins}\n\n"
        "[Output]\n{outp}\n\n"
        "[System]\n{criteria}\n"
    )

    criteria = (
        "Revise the original instruction to improve clarity, reusability, and alignment with the output.\n"
        "- Preserve the intent and task type of the original instruction.\n"
        "- Avoid copying exact wording or facts from the output.\n"
        "- Generalize the instruction so it applies to similar input-output examples.\n"
        "- Improve fluency and remove ambiguity if present.\n"
        "Respond in the following format:\n"
        "[Better Instruction] your revised instruction [End]"
    )

    prompt = prompt_template.format(ins=ins, outp=outp, criteria=criteria)
    return sys_prompt, prompt

def gen_instruction_from_input_output(ins, inp, outp):
    sys_prompt = (
        "You are a helpful assistant for rewriting vague or overly specific instructions "
        "based on the original instruction, input, and output."
    )

    prompt_template = (
        "[Original Instruction]\n{ins}\n\n"
        "[Input]\n{inp}\n\n"
        "[Output]\n{outp}\n\n"
        "[System]\n{criteria}\n"
    )

    criteria = (
        "You are given an original instruction, along with an input-output pair.\n"
        "Revise the instruction to make it clearer and reusable across similar tasks.\n"
        "- Preserve the original instruction's purpose and task type.\n"
        "- Do NOT reuse exact phrases from the input or output.\n"
        "- Improve generalizability and fluency.\n"
        "Respond in the following format:\n"
        "[Better Instruction] your revised instruction [End]"
    )

    prompt = prompt_template.format(ins=ins, inp=inp, outp=outp, criteria=criteria)
    return sys_prompt, prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)  # .jsonl ?? ?? ??
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_model", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--api_base", type=str, default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    # OpenAI API Key ??
    openai.api_key = args.api_key
    if args.api_base:
        openai.api_base = args.api_base

    # ??? ??
    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    # ??? ??? ??
    message_list = []
    token_len_list = []

    for i, data_i in enumerate(sampled_data):
        instruct_i = data_i.get('instruction', '').strip()
        output_raw = data_i.get('output', '')
        if isinstance(output_raw, list):
            output_i = ' '.join(output_raw).strip()
        elif isinstance(output_raw, str):
            output_i = output_raw.strip()
        input_i = data_i.get('input', '').strip() if 'input' in data_i else ''

        whole_text = instruct_i + input_i + output_i
        inputs = llama_tokenizer(whole_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] > 2048:
            gap = input_ids.shape[1] - 2048
            output_i = output_i[:-gap]  # ?? ?? ? ??? ??

        if input_i == '':
            sys_prompt, prompt = gen_instruction_from_output(instruct_i, output_i)
        else:
            sys_prompt, prompt = gen_instruction_from_input_output(instruct_i, input_i, output_i)

        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        message_list.append(message)
        token_len_list.append(len(gpt_encoder.encode(prompt)))

    # JSONL ??? ?? (?? ??)
    batch_size = args.batch_size
    i = 0
    wait_base = 10
    error = 0

    pbar = tqdm(total=len(message_list))
    with open(args.save_path, "w", encoding="utf-8") as f:  # JSONL ?? ??
        while i < len(message_list):
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
                    json.dump(prediction, f, ensure_ascii=False)  # UTF-8 ??? ??
                    f.write("\n")  # JSONL ???? ?? ??
                
                i += batch_size  # ?? ??
                wait_base = 10  # wait ?? ???
                pbar.update(batch_size)  # ??? ????
                pbar.set_description(f"Processing batch {i}/{len(message_list)}")

            except Exception as e:
                error += 1
                logger.error(f"Batch error ({i} - {i+batch_size}): {e}")
                time.sleep(wait_base)
                wait_base *= 2  # ?? ????? ?? ?? ??

    pbar.close()
    logger.info(f"??: ? {len(message_list)}? ?? ? {error}? ??")
