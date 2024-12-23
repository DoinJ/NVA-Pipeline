import argparse
import os.path
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


def eval_QA(args, model_name, file_name, llm):
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
    pos_lst, neg_lst = get_prompt.generate_pos_neg_prompts(data_lst)

    # 计算perplexity，困惑度越小越好
    logging.info("Calculating perplexity")
    pos_ppl = llm.get_perplexity(pos_lst, batch=32)
    neg_ppl = llm.get_perplexity(neg_lst, batch=32)

    right_num = 0
    for i, (row, pos, neg) in enumerate(zip(data_lst, pos_ppl, neg_ppl)):
        row["pos_ppl"] = pos
        row["neg_ppl"] = neg
        if pos < neg:
            row["ppl"] = "right"
            right_num += 1
        else:
            row["ppl"] = "wrong"
    right_rate = right_num / len(data_lst)

    data_lst.insert(0, {
        "model_name": model_name,
        "data_name": args.data_name,
        "right_rate": right_rate,
        "all_number": len(data_lst),
        "right_number": right_num,
        "wrong_number": len(data_lst) - right_num,
    })

    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/") is False:
        os.makedirs(f"res/{args.data_name}/{model_name.split('/')[-1]}/", exist_ok=True)

    output_file_path = f"res/{args.data_name}/{model_name.split('/')[-1]}/QA.json"
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
    eval_QA(args, model_name, file_name, llm)

    file_name = f"data/{args.data_name}/human_statement_result.json"
    eval_QA(args, model_name, file_name, llm)
