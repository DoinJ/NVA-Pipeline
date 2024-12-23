import argparse
import pandas as pd
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
    df = pd.read_csv(f'data/{args.data_name}/sensitive_document.csv')
    data_lst = df['Document'].tolist()
    print(f"data_lst: {data_lst[0]}")

    data_lst = [d[:1920] for d in data_lst if type(d) == str and len(d) > 0]

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    until = tokenizer.eos_token_id
    until = tokenizer.decode(until)
    until = [until, "<|end_of_text|>"]
    print(f"until: {until}")

    # 生成prompt
    all_prompt_ids = get_prompt.generate_prompt_ids(tokenizer, data_lst,args)

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
    outputs = llm.generate(
        # all_prompt,
        prompt_token_ids=all_prompt_ids,  # 这一步之前需要使用模板
        sampling_params=sampling_params,
    )

    # Save the outputs to a json file.
    # 将prompt和result构造为list[dict]
    result_lst = []
    for prompt, output in zip(data_lst, outputs):
        result_lst.append({
            "prompt": prompt,
            "output": output.outputs[0].text,
        })

    output_file_path = f"data/{args.data_name}/statement_result.json"
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(result_lst, f, ensure_ascii=False, indent=4)
