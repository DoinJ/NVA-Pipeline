from pprint import pprint
import kagglehub
from datasets import load_dataset, get_dataset_config_names
import pandas as pd

# diverse_french_news

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/diverse_french_news'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    'gustavecortal/diverse_french_news',
    cache_dir=cache_dir,
    split='train',
)
print(f"diverse_french_news:{dataset}")

# French-PD-Newspapers
'''
cache_dir = './data/French-PD-Newspapers'
dataset = load_dataset(
    'PleIAs/French-PD-Newspapers',
    cache_dir=cache_dir,
    split='train',
)
print(dataset)
dataset = dataset.select(range(3000))  # Select the first 3000 rows
dataset = dataset['complete_text']
print(type(dataset))
# pprint(dataset[:2])
'''

# German-PD-Newspapers

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/German-PD-Newspapers'
dataset = load_dataset(
    'storytracer/German-PD-Newspapers',
    cache_dir=cache_dir,
    split='train',
)
# pprint(dataset[:2])
print(f"German-PD-Newspapers:{dataset}")


# German News Dataset
'''
cache_dir = './data/german-news-dataset'
dataset = pd.read_csv(cache_dir + '/data.csv')
dataset = dataset["text"]
dataset = dataset.head(3000)  # Select the first 3000 rows
dataset = dataset.tolist()  # Convert the Series to a list
# dataset = [item[:3000] if isinstance(item, str) else '' for item in dataset]
pprint(dataset[:1])
print(type(dataset))
'''
'''
cache_dir = './data/German_News_Dataset'
# Download latest version
dataset = kagglehub.dataset_download(
    "pqbsbk/german-news-dataset",
    path = cache_dir
    )
print(dataset)
pprint(dataset[:2])
'''

# bbc_news_alltime

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/bbc_news_alltime'
# 加载数据集，并指定缓存目录
configs = get_dataset_config_names('RealTimeData/bbc_news_alltime')
print(f"bbc_news_alltime:{configs}")
# 下载所有子集并缓存到指定目录
'''
datasets = {}
for config in configs:
    # print(config)
    datasets[config] = load_dataset(
        'RealTimeData/bbc_news_alltime',
        config,
        cache_dir=f"{cache_dir}/{config}",
        split='train',
    )
    # print(f"bbc_news_alltime:{datasets[config]}")
'''

# gov_xuexiqiangguo

# 设置缓存目录
cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/gov_xuexiqiangguo/'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    "liwu/MNBVC",
    'gov_xuexiqiangguo',
    split='train',
    streaming=True,
    cache_dir=cache_dir
)
# print(next(iter(dataset))) # get the first line
print(f"gov_xuexiqiangguo:{dataset}")
# pprint(dataset[:2])


# qa_mfa

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/qa_mfa/'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    "liwu/MNBVC",
    'qa_mfa',
    split='train',
    streaming=True,
    cache_dir=cache_dir
)
print(f"qa_mfa:{dataset}")
# pprint(dataset[:10])


# news_peoples_daily

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/news_peoples_daily/'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    "liwu/MNBVC",
    'news_peoples_daily',
    split='train',
    streaming=True,
    cache_dir=cache_dir
)
print(f"news_peoples_daily:{dataset}")
# pprint(dataset[:10])


# new_york_times_news_2000_2007

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/new_york_times_news_2000_2007'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    'ErikCikalleshi/new_york_times_news_2000_2007',
    cache_dir=cache_dir,
    split='train',
)
print(f"new_york_times_news_2000_2007:{dataset}")
# pprint(dataset[:2])


# cnn

cache_dir = '/aifs4su/hansirui/juchengyi/价值观/1_data/data/cnn'
# 加载数据集，并指定缓存目录
dataset = load_dataset(
    'abisee/cnn_dailymail',
    '3.0.0',
    cache_dir=cache_dir,
    split='train',
)
print(f"cnn:{dataset}")
# pprint(dataset[:1])
# print(type(dataset))
