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

# set up logging
logging.basicConfig(level=logging.INFO)


def eval_statement(args, model_name, file_name, llm):
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
    pos_agree_lst, pos_disagree_lst, neg_agree_lst, neg_disagree_lst, pos_lst, neg_lst = get_prompt.generate_agree_disagree_prompts(
        data_lst)

    # 计算perplexity，困惑度越小越好
    logging.info("Calculating perplexity")
    pos_agree_ppl = llm.get_perplexity(pos_agree_lst, batch=32)
    pos_disagree_ppl = llm.get_perplexity(pos_disagree_lst, batch=32)
    neg_agree_ppl = llm.get_perplexity(neg_agree_lst, batch=32)
    neg_disagree_ppl = llm.get_perplexity(neg_disagree_lst, batch=32)

    pos_right_num = 0
    neg_right_num = 0
    for i, (row, agree, disagree) in enumerate(zip(pos_lst, pos_agree_ppl, pos_disagree_ppl)):
        row["agree_ppl"] = agree
        row["disagree_ppl"] = disagree
        if agree < disagree:
            row["ppl"] = "pos_right"
            pos_right_num += 1
        else:
            row["ppl"] = "pos_wrong"
    pos_right_rate = pos_right_num / len(pos_agree_ppl)

    for i, (row, agree, disagree) in enumerate(zip(neg_lst, neg_agree_ppl, neg_disagree_ppl)):
        row["agree_ppl"] = agree
        row["disagree_ppl"] = disagree
        if disagree < agree:
            row["ppl"] = "neg_right"
            neg_right_num += 1
        else:
            row["ppl"] = "neg_wrong"

    data_lst.insert(0, {
        "model_name": model_name,
        "data_name": args.data_name,
        "total_right_rate": (pos_right_num + neg_right_num) / len(data_lst),
        "pos_right_rate": pos_right_rate,
        "neg_right_rate": neg_right_num / len(neg_agree_ppl),
        "all_number": len(data_lst),
        "pos_right_number": pos_right_num,
        "neg_right_number": neg_right_num,
        "pos_number": len(pos_agree_ppl),
        "neg_number": len(neg_agree_ppl),
    })

    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/") is False:
        os.makedirs(f"res/{args.data_name}/{model_name.split('/')[-1]}/", exist_ok=True)
    output_file_path = f"res/{args.data_name}/{model_name.split('/')[-1]}/statement_agree.json"
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
    llm = lmppl.LM(model_name, torch_dtype=torch.bfloat16, num_gpus=1,
                   use_auth_token="")
    eval_statement(args, model_name, file_name, llm)

    file_name = f"data/{args.data_name}/human_statement_result.json"
    eval_statement(args, model_name, file_name, llm)
