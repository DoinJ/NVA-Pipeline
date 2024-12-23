import json
import logging
import os
from typing import List

from datasets import Dataset, get_dataset_config_names, load_dataset

from tqdm import tqdm

# Set the logging level
logging.basicConfig(level=logging.INFO)

def my_load_dataset(data_name: str) -> List[str]:
    """
    This function loads a dataset from a directory containing JSON files.
    It iterates over all the files in the directory, loads the JSON data, and concatenates it into a single dataset.

    Returns:
        dataset: A Dataset object containing the concatenated data from all the JSON files.
    """

    # Initialize an empty list to store the data from each file
    data_list = []
    if data_name in ["appledaily", "pressreleases"]:
        if data_name == "appledaily":
            file_path = "data/appledaily_clean_json/"
            # Iterate over all the files in the directory
            for file in tqdm(
                    os.listdir(file_path),  # Directory to load files from
                    desc="Loading dataset",  # Description to display on the progress bar
                    total=len(os.listdir(file_path))  # Total number of files to load
            ):
                # Load the JSON data from the current file
                data = json.load(open(f"{file_path}{file}", "r", encoding="utf-8"))
                # Append the data to the list
                data_list.extend(data)
        elif data_name == "pressreleases":
            file_path = "data/pressreleases_clean_json/"
            # using walk through, and iterate over all the files in the directory
            # 递归遍历，遇到文件夹继续遍历
            for root, dirs, files in os.walk(file_path):
                for file in tqdm(
                        files,  # Directory to load files from
                        desc="Loading dataset",  # Description to display on the progress bar
                        total=len(files)  # Total number of files to load
                ):
                    data = json.load(open(os.path.join(root, file), "r", encoding="utf-8"))
                    # Append the data to the list
                    data_list.extend(data)
        # Concatenate the data from all the files into a single dataset
        dataset = Dataset.from_list(data_list)

        # Log the number of samples in the dataset
        logging.info(f"Loaded dataset with {len(dataset)} samples")
        # Log the column names in the dataset
        logging.info(f"Columns: {dataset.column_names}")
        # Log the dataset
        logging.info(f"data: {dataset}")

        # Extract the 'content' column from the dataset
        dataset = dataset["content"]

        return dataset

    elif data_name == "gov_xuexiqiangguo":
        cache_dir = './data/gov_xuexiqiangguo/'
        dataset = load_dataset(
            "liwu/MNBVC",
            'gov_xuexiqiangguo',
            split='train',
            cache_dir=cache_dir
        )
        dataset = dataset["段落"]
        text_lst = []
        for page in dataset:
            s = []
            for line in page:
                s.append(line["内容"])
            text_lst.append("".join(s))
        text_lst = [item[:3000] for item in text_lst]
        return text_lst

    elif data_name == "zh_mfa":
        cache_dir = './data/qa_mfa/'
        dataset = load_dataset(
            "liwu/MNBVC",
            'qa_mfa',
            split='train',
            cache_dir=cache_dir
        )
        dataset = dataset["答"]
        dataset = [item[:3000] for item in dataset]
        return dataset

    elif data_name == "news_peoples_daily":
        cache_dir = './data/news_peoples_daily/'
        dataset = load_dataset(
            "liwu/MNBVC",
            'news_peoples_daily',
            split='train',
            cache_dir=cache_dir
        )
        dataset = dataset["段落"]
        text_lst = []
        for page in dataset:
            s = []
            for line in page:
                s.append(line["内容"])
            text_lst.append("".join(s))
        text_lst = [item[:3000] for item in text_lst]
        return text_lst

    elif data_name == "bbc":
        # 加载数据集，并指定缓存目录
        configs = get_dataset_config_names('RealTimeData/bbc_news_alltime')
        datasets = {}
        for config in configs:
            datasets[config] = load_dataset(
                'RealTimeData/bbc_news_alltime',
                config,
                cache_dir=f"./data/bbc_news_alltime/{config}",
                split='train',
            )
        text_lst = []
        for config in configs:
            for dataset in datasets[config]:
                s = dataset["content"]
                text_lst.append(s)
        text_lst = [item[:3000] for item in text_lst]
        return text_lst

    elif data_name == "cnn":
        # 加载数据集，并指定缓存目录
        cache_dir = './data/cnn/'
        dataset = load_dataset(
            'abisee/cnn_dailymail',
            '3.0.0',
            cache_dir=cache_dir,
            split='train',
        )
        dataset = dataset["highlights"]
        dataset = [item[:3000] for item in dataset]
        return dataset

    elif data_name == "new_york_times":
        cache_dir = './data/new_york_times_news_2000_2007'
        dataset = load_dataset(
            'ErikCikalleshi/new_york_times_news_2000_2007',
            split='train',
            cache_dir=cache_dir
        )
        dataset = dataset["content"]
        dataset = [item[:3000] for item in dataset]
        return dataset



