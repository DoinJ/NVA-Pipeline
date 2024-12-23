import argparse
import os.path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import get_prompt
import logging
import gc
from vllm import LLM, SamplingParams
from evaluate import load
import multiprocessing as mp

# set up logging
logging.basicConfig(level=logging.INFO)

import pynvml
def get_gpu_memory_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # Assuming you're using the first GPU (index 1)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024  # Convert bytes to megabytes

def calculate_perplexity(model_name, positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst, result_queue):
    device_ids = [1,2]  # Specify the desired device IDs
    devices = [torch.device(f"cuda:{i}") for i in device_ids]
    # device = torch.device("cuda:1")  # Specify the CUDA device with ID 0
    for device in devices:
        with torch.cuda.device(device):
            try:
                perplexity = load("perplexity", module_type="metric")
                pos_A_ppl = perplexity.compute(model_id=model_name, batch_size=8,add_start_token=False, predictions=positive_A_lst)["perplexities"]
                pos_B_ppl = perplexity.compute(model_id=model_name, batch_size=8,add_start_token=False, predictions=positive_B_lst)["perplexities"]
                neg_A_ppl = perplexity.compute(model_id=model_name, batch_size=8,add_start_token=False, predictions=negative_A_lst)["perplexities"]
                neg_B_ppl = perplexity.compute(model_id=model_name, batch_size=8,add_start_token=False, predictions=negative_B_lst)["perplexities"]
                result_queue.put((pos_A_ppl, pos_B_ppl, neg_A_ppl, neg_B_ppl))
            finally:
                del perplexity
                torch.cuda.empty_cache()

def eval_QA(args, model_name, file_name, v_llm, flag): # llm, directly use model_name as llm
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
    questions = []
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
        questions.append(row["generated Q"])

    logging.info(f"len of data_lst: {len(data_lst)}")

    question_ids = get_prompt.get_ids(tokenizer, questions)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=512,
        stop=until,
    )
    print(f"GPU memory usage: {get_gpu_memory_usage()} MB")
    # 查看输出结果
    logging.info("Generating")
    outputs = v_llm.generate(
        # all_prompt,
        prompt_token_ids=question_ids,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )
    output_texts = [output.outputs[0].text for output in outputs]   

    # Free GPU memory
    del v_llm
    del outputs 
    gc.collect()
    torch.cuda.empty_cache()  
    print(f"GPU memory usage: {get_gpu_memory_usage()} MB")
    # 生成prompt
    positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst = get_prompt.generate_judge_cosistency(data_lst,output_texts)

    # 计算perplexity，困惑度越小越好
    logging.info("Calculating perplexity")
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    perplexity_process = ctx.Process(target=calculate_perplexity, args=(
        model_name, positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst, result_queue
    ))
    perplexity_process.start()
    pos_A_ppl, pos_B_ppl, neg_A_ppl, neg_B_ppl = result_queue.get()
    perplexity_process.join()
    '''
    perplexity = load("perplexity", module_type="metric")
    pos_A_ppl = perplexity.compute(model_id=model_name, add_start_token=False, predictions=positive_A_lst)["perplexities"]
    pos_B_ppl = perplexity.compute(model_id=model_name, add_start_token=False, predictions=positive_B_lst)["perplexities"]
    neg_A_ppl = perplexity.compute(model_id=model_name, add_start_token=False, predictions=negative_A_lst)["perplexities"]
    neg_B_ppl = perplexity.compute(model_id=model_name, add_start_token=False, predictions=negative_B_lst)["perplexities"]
    '''
    right_num = 0
    pos_right_num = 0
    neg_right_num = 0
    for i, (row, A, B) in enumerate(zip(data_lst[:len(data_lst) // 2], pos_A_ppl, pos_B_ppl)):
        row["A_ppl"] = A
        row["B_ppl"] = B
        if A < B:
            row["ppl"] = "right"
            right_num += 1
            pos_right_num += 1
        else:
            row["ppl"] = "wrong"

    for i, (row, A, B) in enumerate(zip(data_lst[len(data_lst) // 2:], neg_A_ppl, neg_B_ppl)):
        row["A_ppl"] = A
        row["B_ppl"] = B
        if A > B:
            row["ppl"] = "right"
            right_num += 1
            neg_right_num += 1
        else:
            row["ppl"] = "wrong"

    data_lst.insert(0, {
        "model_name": model_name.split('/')[-1],
        "data_name": args.data_name,
        "right_rate": right_num / len(data_lst),
        "all_number": len(data_lst),
        "right_number": right_num,
        "wrong_number": len(data_lst) - right_num,
        "pos_right_rate": pos_right_num / len(data_lst) // 2,
        "pos_all_number": len(data_lst) // 2,
        "pos_right_number": pos_right_num,
        "pos_wrong_number": len(data_lst) // 2 - pos_right_num,
        "neg_right_rate": neg_right_num / len(data_lst) // 2,
        "neg_all_number": len(data_lst) // 2,
        "neg_right_number": neg_right_num,
        "neg_wrong_number": len(data_lst) // 2 - neg_right_num,
    })

    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/") is False:
        os.makedirs(f"res/{args.data_name}/{model_name.split('/')[-1]}/", exist_ok=True)

    output_file_path = f"res/{args.data_name}/{model_name.split('/')[-1]}/{flag}_cosistency.json"
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(data_lst, f, ensure_ascii=False, indent=4)


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

    # llm = AutoModelForCausalLM.from_pretrained(model_name)
    # llm = lmppl.LM(model_name, torch_dtype=torch.bfloat16, num_gpus=1,
                   # use_auth_token="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")
    v_llm = LLM(
        # enforce_eager=True,
        model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        # tensor_parallel_size=8
    )
    file_name = f"data/{args.data_name}/official_statement_result.json"
    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/official_cosistency.json") is False:
        eval_QA(args, model_name, file_name, v_llm, flag="official")

    file_name = f"data/{args.data_name}/human_statement_result.json"
    if os.path.exists(f"res/{args.data_name}/{model_name.split('/')[-1]}/human_cosistency.json") is False:
        eval_QA(args, model_name, file_name, v_llm, flag="human")
