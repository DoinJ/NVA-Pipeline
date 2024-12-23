import argparse
import os.path
import random

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import get_prompt
import logging

# set up logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # create a parser object
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--data_name", type=str, default="pressreleases",
                        choices=[
                            "pressreleases",
                            "gov_xuexiqiangguo", "zh_mfa", "news_peoples_daily",
                            "bbc",
                            "cnn",
                            "new_york_times",
                            "french",
                            "german"
                        ])
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

    args = parser.parse_args()
    model_name = args.llm_name

    # 读取数据
    df = pd.read_json(f"data/{args.data_name}/source_statement_result.json")
    logging.info(f"len of df: {df.shape[0]}")
    data_lst = []
    human_data_lst = []
    for idx, row in df.iterrows():
        raw_news = row["raw_news"]
        statement = row["statement"]
        source = row["source"]
        if ("unknown" in source):
            # 官方媒体的观点
            data_lst.append({
                "raw_news": raw_news,
                "statement": statement,
                "filter_flag": row["filter_flag"],
                "full_source": row["full_source"],
                "source": source,
            })
        else:
            # 来自其他人的观点
            human_data_lst.append({
                "raw_news": raw_news,
                "statement": statement,
                "filter_flag": row["filter_flag"],
                "full_source": row["full_source"],
                "source": source,
            })

    logging.info(f"len of data_lst: {len(data_lst)}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    until = tokenizer.eos_token_id
    until = tokenizer.decode(until)
    until = [until, "<|end_of_text|>"]
    print(f"until: {until}")

    # 生成prompt
    negative_prompt_ids = get_prompt.generate_negative_prompt_ids(tokenizer, data_lst)
    negative_human_prompt_ids = get_prompt.generate_negative_prompt_ids(tokenizer, human_data_lst)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        # top_k=5,
        # repetition_penalty=1.1,
        max_tokens=2048,
        # min_tokens=20,
        stop=until,
    )

    # Create an LLM.
    llm = LLM(
        # enforce_eager=True,
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        # tensor_parallel_size=8
    )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # 限制gpu内存使用百分比
    negative_outputs = llm.generate(
        # all_prompt,
        prompt_token_ids=negative_prompt_ids,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )
    negative_human_outputs = llm.generate(
        # all_prompt,
        prompt_token_ids=negative_human_prompt_ids,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )

    # Save the outputs to a json file.
    # 将prompt和result构造为list[dict]
    offical_result_lst = []
    for row, output in zip(data_lst, negative_outputs):
        output_text = output.outputs[0].text
        # print(f"output_text: {output_text}")
        if "##" not in output_text:
            # print("## not in output_text")
            continue
        try:
            # 找到 Q: 的位置
            q_loc = output_text.find("Q:")
            # update the output_text
            output_text = output_text[q_loc:]
            lst = output_text.split("##")
            q = lst[0].strip()
            s = lst[1].strip()
            rs = lst[2].strip()
            if not q.startswith("Q:") or not s.startswith("S:") or not rs.startswith("RS:"):
                # print("not start with Q: or S: or RS:")
                continue
            q = q.strip("Q:").strip()
            s = s.strip("S:").strip()
            rs = rs.strip("RS:").strip()
            offical_result_lst.append({
                "raw_news": row["raw_news"],
                "statement": row["statement"],
                "filter_flag": row["filter_flag"],
                "full_source": row["full_source"],
                "source": row["source"],
                "generated Q": q,
                "generated reverse statement": rs,
                "generated raw data": output_text,
            })
        except Exception as e:
            print(e)
            continue

    output_file_path = f"data/{args.data_name}/official_statement_result.json"
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(offical_result_lst, f, ensure_ascii=False, indent=4)

    # Get human data
    human_result_lst = []
    for row, output in zip(human_data_lst, negative_human_outputs):
        output_text = output.outputs[0].text
        if "##" not in output_text:
            # print("## not in output_text")
            continue
        try:
            # 找到 Q: 的位置
            q_loc = output_text.find("Q:")
            # update the output_text
            output_text = output_text[q_loc:]
            lst = output_text.split("##")
            q = lst[0].strip()
            s = lst[1].strip()
            rs = lst[2].strip()
            if not q.startswith("Q:") or not s.startswith("S:") or not rs.startswith("RS:"):
                # print("not start with Q: or S: or RS:")
                continue
            q = q.strip("Q:").strip()
            s = s.strip("S:").strip()
            rs = rs.strip("RS:").strip()
            human_result_lst.append({
                "raw_news": row["raw_news"],
                "statement": row["statement"],
                "filter_flag": row["filter_flag"],
                "full_source": row["full_source"],
                "source": row["source"],
                "generated Q": q,
                "generated reverse statement": rs,
                "generated raw data": output_text,
            })
        except Exception as e:
            print(e)
            continue

    # Save human data
    output_file_path = f"data/{args.data_name}/human_statement_result.json"
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(human_result_lst, f, ensure_ascii=False, indent=4)