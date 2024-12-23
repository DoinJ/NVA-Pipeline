import argparse
import os.path
import random

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import logging

# set up logging
logging.basicConfig(level=logging.INFO)


def get_data(data_name: str, flag: str):
    # 读取数据
    if flag == "official":
        df = pd.read_json(f"../5_正反例构造/data/{data_name}/official_statement_result.json")
    elif flag == "human":
        df = pd.read_json(f"../5_正反例构造/data/{data_name}/human_statement_result.json")
    else:
        raise ValueError(f"Invalid flag: {flag}")
    logging.info(f"len of df: {df.shape[0]}")
    data_lst = []

    for idx, row in df.iterrows():
        statement = row["statement"]
        Q = row["generated Q"]
        RS = row["generated reverse statement"]
        # [
        #   {
        #     "instruction": "human instruction (required)",
        #     "input": "human input (optional)",
        #     "chosen": "chosen answer (required)",
        #     "rejected": "rejected answer (required)"
        #   }
        # ]
        data_lst.append(
            {
                "instruction": Q,
                "input": "",
                "chosen": statement,
                "rejected": RS,
            }
        )

    logging.info(f"len of data_lst: {len(data_lst)}")
    return data_lst


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

    # 读取数据
    data_lst = get_data(args.data_name, "official")
    random.shuffle(data_lst)
    official_train_data_lst = data_lst[:int(len(data_lst) * 0.8)]
    official_test_data_lst = data_lst[int(len(data_lst) * 0.8):]
    for item_dic in official_test_data_lst:
        item_dic["source"] = "official"

    data_lst = get_data(args.data_name, "human")
    random.shuffle(data_lst)
    human_train_data_lst = data_lst[:int(len(data_lst) * 0.8)]
    human_test_data_lst = data_lst[int(len(data_lst) * 0.8):]
    for item_dic in human_test_data_lst:
        item_dic["source"] = "human"

    train_lst = official_train_data_lst + human_train_data_lst
    test_lst = official_test_data_lst + human_test_data_lst

    # Save data_lst to a JSON file
    if not os.path.exists(f"./data/{args.data_name}"):
        os.makedirs(f"./data/{args.data_name}")
    with open(f"./data/{args.data_name}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_lst, f, ensure_ascii=False, indent=4)
    with open(f"./data/{args.data_name}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_lst, f, ensure_ascii=False, indent=4)
