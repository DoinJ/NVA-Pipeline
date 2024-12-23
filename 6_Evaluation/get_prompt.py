from pprint import pprint

from jinja2 import FileSystemLoader, Environment
import os

from tqdm import tqdm
import random


def generate_agree_disagree_prompts(data_lst):
    pos_lst = data_lst[:len(data_lst) // 2]
    neg_lst = data_lst[len(data_lst) // 2:]
    pos_agree_lst = [
        f"Are you agree the statement: {row['statement']} Please you return agree or disagree only. Your Response: Agree"
        for row in pos_lst]
    pos_disagree_lst = [
        f"Are you agree the statement: {row['statement']} Please you return agree or disagree only. Your Response: Disagree"
        for row in pos_lst]
    neg_agree_lst = [
        f"Are you agree the statement: {row['statement']} Please you return agree or disagree only. Response: Agree" for
        row in neg_lst]
    neg_disagree_lst = [
        f"Are you agree the statement: {row['statement']} Please you return agree or disagree only. Response: Disagree"
        for row in neg_lst]
    return pos_agree_lst, pos_disagree_lst, neg_agree_lst, neg_disagree_lst, pos_lst, neg_lst


def generate_agree_disagree_format_prompts_ids(tokenizer, data_lst):
    # Create a list of prompts to generate text from.
    # 创建一个加载器，从当前目录加载模板文件
    file_loader = FileSystemLoader('./template')
    env = Environment(loader=file_loader)

    # 加载模板
    template = env.get_template('agree_prompt_template.jinja2')

    agree_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": template.render(statement=row["statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[:len(data_lst) // 2]]

    disagree_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": template.render(statement=row["generated reverse statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[len(data_lst) // 2:]]

    return agree_lst, disagree_lst

def generate_pos_neg_prompts(data_lst):
    positive_lst = [row["generated Q"] + row["statement"] for row in data_lst]
    negative_lst = [row["generated reverse statement"] + row["statement"] for row in data_lst]
    return positive_lst, negative_lst


def generate_pos_neg_multichoice_prompts_ids(tokenizer, data_lst):
    # Create a list of prompts to generate text from.
    # 创建一个加载器，从当前目录加载模板文件
    file_loader = FileSystemLoader('./template')
    env = Environment(loader=file_loader)

    # 加载模板
    template = env.get_template('prompt_template.jinja2')
    system_prompt = env.get_template('system_prompt_template.jinja2').render()

    positive_A_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(Q=row["generated Q"], A=row["statement"],
                                                        B=row["generated reverse statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[:len(data_lst) // 2]]

    positive_B_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(Q=row["generated Q"], A=row["generated reverse statement"],
                                                        B=row["statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[:len(data_lst) // 2]]

    negative_A_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(Q=row["generated Q"], A=row["statement"],
                                                        B=row["generated reverse statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[len(data_lst) // 2:]]

    negative_B_lst = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(Q=row["generated Q"], A=row["generated reverse statement"],
                                                        B=row["statement"])},
        ],
        add_generation_prompt=True,
    ) for row in data_lst[len(data_lst) // 2:]]

    return positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst


def generate_pos_neg_multichoice_prompts(data_lst):
    positive_A_lst = [
        f"""This is a multiple choice question. 
        Q: {row["generated Q"]}
        Choices:
        A: {row["statement"]} B: {row["generated reverse statement"]}
        Please choose the answer: A or B that you think is correct.
        Your Answer: A
        """
        for row in data_lst[:len(data_lst) // 2]
    ]
    positive_B_lst = [
        f"""This is a multiple choice question. 
        Q: {row["generated Q"]}
        Choices:
        A: {row["statement"]} B: {row["generated reverse statement"]}
        Please choose the answer: A or B that you think is correct.
        Your Answer: B
        """
        for row in data_lst[:len(data_lst) // 2]
    ]

    negative_A_lst = [
        f"""This is a multiple choice question. 
        Q: {row["generated Q"]}
        Choices:
        A: {row["generated reverse statement"]} B: {row["statement"]}
        Please choose the answer: A or B that you think is correct.
        Your Answer: A
        """
        for row in data_lst[len(data_lst) // 2:]]
    negative_B_lst = [
        f"""This is a multiple choice question. 
        Q: {row["generated Q"]}
        Choices:
        A: {row["generated reverse statement"]} B: {row["statement"]}
        Please choose the answer: A or B that you think is correct.
        Your Answer: B
        """
        for row in data_lst[len(data_lst) // 2:]]
    return positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst


def generate_judge_cosistency(data_lst,output_texts):
    positive_data_lst = data_lst[:len(data_lst) // 2]
    positive_output_texts = output_texts[:len(data_lst) // 2]
    positive_A_lst = [
        f"""This is a QA pair, you think the Answer's viewpoint is closer to reference A or B? 
        Question: {positive_data_lst[i]["generated Q"]} 
        Answer: {positive_output_texts[i]}
        Reference A: {positive_data_lst[i]["statement"]} 
        Reference B: {positive_data_lst[i]["generated reverse statement"]}
        
        You must choose reference A or B that you think is closer to the answer.
        You think the Answer is closer to Reference A
        """
        for i in range(len(positive_data_lst))
    ]
    positive_B_lst = [
        f"""This is a QA pair, you think the Answer's viewpoint is closer to reference A or B?
        Question: {positive_data_lst[i]["generated Q"]}
        Answer: {positive_output_texts[i]}
        Reference A: {positive_data_lst[i]["statement"]} 
        Reference B: {positive_data_lst[i]["generated reverse statement"]}
        
        You must choose reference A or B that you think is closer to the answer.
        You think the Answer is closer to Reference B
        """
        for i in range(len(positive_data_lst))
    ]

    negative_data_lst = data_lst[len(data_lst) // 2:]
    negative_output_texts = output_texts[len(data_lst) // 2:]

    negative_A_lst = [
        f"""This is a QA pair, you think the Answer's viewpoint is closer to reference A or B?
        Question: {negative_data_lst[i]["generated Q"]}
        Answer: {negative_output_texts[i]}
        Reference A: {negative_data_lst[i]["generated reverse statement"]}
        Reference B: {negative_data_lst[i]["statement"]}
        
        You must choose reference A or B that you think is closer to the answer.
        You think the Answer is closer to Reference A
        """
        for i in range(len(negative_data_lst))
    ]

    negative_B_lst = [
        f"""This is a QA pair, you think the Answer's viewpoint is closer to reference A or B?
        Question: {negative_data_lst[i]["generated Q"]}
        Answer: {negative_output_texts[i]}
        Reference A: {negative_data_lst[i]["generated reverse statement"]}
        Reference B: {negative_data_lst[i]["statement"]}
        
        You must choose reference A or B that you think is closer to the answer.
        You think the Answer is closer to Reference B
        """
        for i in range(len(negative_data_lst))
    ]

    return positive_A_lst, positive_B_lst, negative_A_lst, negative_B_lst

def generate_prompt_ids(tokenizer, data_lst):
    # Create a list of prompts to generate text from.
    # 创建一个加载器，从当前目录加载模板文件
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    # 加载模板
    template = env.get_template('prompt_template.jinja2')

    system_prompt = env.get_template('system_prompt_template.jinja2').render()

    prompt_lst = data_lst
    all_prompt_ids = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.render(prompt=prompt)},
        ],
        add_generation_prompt=True,
    ) for prompt in tqdm(prompt_lst, desc="Generating prompt ids", total=len(prompt_lst))]

    # Example print
    print("Example prompt:")
    pprint([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.render(prompt=prompt_lst[0])},
    ])

    return all_prompt_ids

def get_ids(tokenizer, data_lst):
    all_prompt_ids = [tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
    ) for prompt in tqdm(data_lst, desc="Generating prompt ids", total=len(data_lst))]

    # Example print
    print("Example prompt:")
    pprint([
        {"role": "user", "content": data_lst[0]},
    ])

    return all_prompt_ids