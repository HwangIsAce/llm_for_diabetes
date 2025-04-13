import json
import openai
import string
import argparse
import re

def gen_instruction_prompt_no_input(ins, outp):
    sys_prompt = (
        "You are a helpful and strict assistant for improving unclear or vague instructions.\n"
        "Revise the instruction to better match the provided output."
    )

    prompt_template = (
        "[Instruction]\n{ins}\n\n"
        "[Output]\n{outp}\n\n"
        "[System]\n{criteria}\n"
    )

    criteria = (
        "We would like you to revise the given instruction based on the output.\n"
        "1. Why is the instruction vague, incomplete, or misaligned with the output?\n"
        "2. Rewrite it to be clear, specific, and properly aligned.\n"
        "Respond in this format:\n[Better Instruction] your improved instruction [End]"
    )

    prompt = prompt_template.format(ins=ins, outp=outp, criteria=criteria)
    return sys_prompt, prompt


def gen_instruction_prompt_input(ins, inp, outp):
    sys_prompt = (
        "You are a helpful and strict assistant for improving vague or insufficient instructions.\n"
        "Revise the instruction to better reflect the provided input and output."
    )

    prompt_template = (
        "[Instruction]\n{ins}\n\n"
        "[Input]\n{inp}\n\n"
        "[Output]\n{outp}\n\n"
        "[System]\n{criteria}\n"
    )

    criteria = (
        "We would like you to revise the given instruction based on the input and output.\n"
        "1. Why is the instruction insufficient, ambiguous, or not aligned with the input/output?\n"
        "2. Rewrite it to be specific, well-structured, and well-aligned.\n"
        "Respond in this format:\n[Better Instruction] your improved instruction [End]"
    )

    prompt = prompt_template.format(ins=ins, inp=inp, outp=outp, criteria=criteria)
    return sys_prompt, prompt



def extract_segments(text):
    if text.count('[Better Instruction]') >= 2:
        pattern = r'\[(Better Instruction)\](.*?)(\[End\]|\[Better Instruction\]|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    else:
        # pattern = r'\[(Better Instruction)\](.*?)\[End\]'
        pattern = r'\[(Better Instruction)\](.*?)(\[End\]|End|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    return [segment[1].strip() for segment in segments]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default='')
    parser.add_argument("--ori_data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--save_intermediate_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="maximum number of tokens produced in the output",
    )

    args = parser.parse_args()
    openai.api_key = args.api_key
    model_engine = args.api_model

    with open(args.raw_data_path,'r') as f:
        raw_data = json.load(f)

    with open(args.ori_data_path,'r') as f:
        ori_data = json.load(f)

    new_data = []
    for i, raw_data_i in enumerate(raw_data):
        if (i+1) % 1000 == 0:
            print(i+1,'/',len(raw_data))
        seg_list = extract_segments(raw_data_i)

        ori_data_i = ori_data[i]
        instruct_i = ori_data_i['instruction'].strip()
        output_raw = ori_data_i['output']
        if isinstance(output_raw, list):
           output_i = ' '.join(output_raw).strip()
        elif isinstance(output_raw, str):
            output_i = ori_data_i['output'].strip() 
        if 'input' in ori_data_i.keys():
            input_i = ori_data_i['input'].strip()
        else:
            input_i = ''

        if len(seg_list) != 1:

            if input_i == '':
                sys_prompt, prompt = gen_instruction_prompt_no_input(instruct_i, output_i)
            else:
                sys_prompt, prompt = gen_instruction_prompt_input(instruct_i, input_i, output_i)
            response = ''

            try:
                message =[
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                ]
                completion = openai.ChatCompletion.create(
                            model=model_engine,
                            messages=message,
                            temperature=0.0,
                            max_tokens=2048,
                            top_p=1.0,
                )
                response = completion.choices[0].message.content
            except:
                response = ''

            seg_list = extract_segments(response)
            pass

        if len(seg_list) != 1:
            seg_list = ['']
    
        temp_data = {}
        temp_data['instruction'] = ori_data_i['instruction']
        temp_data['output'] = ori_data_i['output']
        temp_data['input'] = input_i
        temp_data['better_instruction'] = seg_list[0]
        new_data.append(temp_data)


    if args.save_intermediate_path != '':
        with open(args.save_intermediate_path,'w') as f:
            json.dump(new_data,f,indent=4)

    final_new_data = []
    none_count = 0
    for i, data_i in enumerate(new_data):

        temp_data = {}
        temp_data['instruction'] = data_i['better_instruction']
        temp_data['input'] = data_i['input']
        temp_data['output'] = data_i['output']
        final_new_data.append(temp_data)

    print('none_num',none_count)
    print('Len New Data', len(final_new_data))
    with open(args.save_path,'w') as f:
        json.dump(final_new_data,f,indent=4)