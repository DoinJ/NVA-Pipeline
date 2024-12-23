import argparse
import os.path
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import get_prompt
import logging
import lmppl
from vllm import LLM, SamplingParams

# set up logging
logging.basicConfig(level=logging.INFO)


def eval_statement(args, model_name, file_name, llm, flag):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    until = tokenizer.eos_token_id
    until = tokenizer.decode(until)
    until = [until, "<|end_of_text|>"]
    print(f"until: {until}")

    # 读取数据
    df = pd.read_json(file_name)
    logging.info(f"len of df: {df.shape[0]}")
    data_lst = []
    for idx, row in df.iterrows():
        data_lst.append({
            "raw_news": row["raw_news"],
            "statement": row["statement"],
            "filter_flag": row["filter_flag"],
            "full_source": row["full_source"],
            "source": row["source"],
            "generated Q": row["generated Q"],
            "generated reverse statement": row["generated reverse statement"],
            "generated raw data": row["generated raw data"],
        })

    logging.info(f"len of data_lst: {len(data_lst)}")
    # 生成prompt
    random.seed(42)
    random.shuffle(data_lst)
    agree_lst, disagree_lst = get_prompt.generate_agree_disagree_format_prompts_ids(
        tokenizer,
        data_lst
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=512,
        stop=until,
    )

    # 查看输出结果
    logging.info("Generating")
    agree_outputs = llm.generate(
        # all_prompt,
        prompt_token_ids=agree_lst,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )
    disagree_outputs = llm.generate(
        # all_prompt,
        prompt_token_ids=disagree_lst,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )

    right_num = 0
    pos_right_num = 0
    neg_right_num = 0
    for i, (row, agree) in enumerate(zip(data_lst[:len(data_lst) // 2], agree_outputs)):
        output_text = agree.outputs[0].text
        row["agree_response"] = output_text
        text_lst = output_text.split()
        if "agree" in text_lst and "disagree" not in text_lst:
            row["ppl"] = "pos_right"
            pos_right_num += 1
            right_num += 1
        else:
            row["ppl"] = "pos_wrong"
    pos_right_rate = pos_right_num / len(data_lst[:len(data_lst) // 2])

    for i, (row, disagree) in enumerate(zip(data_lst[len(data_lst) // 2:], disagree_outputs)):
        output_text = disagree.outputs[0].text
        row["disagree_response"] = output_text
        text_lst = output_text.split()
        if "disagree" in text_lst and "agree" not in text_lst:
            row["ppl"] = "neg_right"
            neg_right_num += 1
            right_num += 1
        else:
            row["ppl"] = "neg_wrong"

    data_lst.insert(0, {
        "model_name": model_name,
        "data_name": args.data_name,
        "total_right_rate": (pos_right_num + neg_right_num) / len(data_lst),
        "pos_right_rate": pos_right_rate,
        "neg_right_rate": neg_right_num / len(data_lst[len(data_lst) // 2:]),
        "all_number": len(data_lst),
        "pos_right_number": pos_right_num,
        "neg_right_number": neg_right_num,
        "pos_number": len(data_lst) // 2,
        "neg_number": len(data_lst) // 2,
    })

    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/") is False:
        os.makedirs(f"res/{args.data_name}/{model_name.split('/')[-1]}/", exist_ok=True)
    output_file_path = f"res/{args.data_name}/{model_name.split('/')[-1]}/{flag}_statement_agree_format_output.json"
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(data_lst, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # create a parser object
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--data_name", type=str, default="appledaily",
                        choices=[
                            "appledaily", "pressreleases",
                            "gov_xuexiqiangguo", "zh_mfa", "news_peoples_daily",
                            "bbc",
                            "cnn",
                            "new_york_times",
                        ])
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

    args = parser.parse_args()
    model_name = args.llm_name

    file_name = f"data/{args.data_name}/official_statement_result.json"
    llm = LLM(
        # enforce_eager=True,
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        # tensor_parallel_size=8

    )
    eval_statement(args, model_name, file_name, llm, flag="official")

    file_name = f"data/{args.data_name}/human_statement_result.json"
    eval_statement(args, model_name, file_name, llm, flag="human")
