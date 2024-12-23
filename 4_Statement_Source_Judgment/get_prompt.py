from pprint import pprint

from jinja2 import FileSystemLoader, Environment
import os

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_prompt_ids(tokenizer, data_lst):
    # Create a list of prompts to generate text from.
    # 创建一个加载器，从当前目录加载模板文件
    file_loader = FileSystemLoader('./template')
    env = Environment(loader=file_loader)

    # 加载模板
    template = env.get_template('prompt_template.jinja2')
    system_prompt = env.get_template('system_prompt_template.jinja2').render()

    prompt_lst = [row["statement"] for row in data_lst]
    all_prompt_ids = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(prompt=prompt)},
        ],
        add_generation_prompt=True,
    ) for prompt in tqdm(prompt_lst, desc="Generating prompt ids",total=len(prompt_lst))]

    # Example print
    print("Example prompt:")
    pprint([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.render(prompt=prompt_lst[0])},
    ])

    return all_prompt_ids