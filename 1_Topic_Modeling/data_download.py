from pprint import pprint

from datasets import load_dataset, get_dataset_config_names

cache_dir = './data/bbc_news_alltime'
# # 加载数据集，并指定缓存目录
configs = get_dataset_config_names('RealTimeData/bbc_news_alltime')
# 下载所有子集并缓存到指定目录
datasets = {}
for config in configs:
    print(config)
    datasets[config] = load_dataset(
        'RealTimeData/bbc_news_alltime',
        config,
        cache_dir=f"{cache_dir}/{config}",
        split='train',
    )
    print(datasets[config])

'''
# 设置缓存目录
cache_dir = './data/gov_xuexiqiangguo/'

# 加载数据集，并指定缓存目录
dataset = load_dataset(
    "liwu/MNBVC",
    'gov_xuexiqiangguo',
    split='train',
    # streaming=True,
    cache_dir=cache_dir
)
# print(next(iter(dataset))) # get the first line
print(dataset)
pprint(dataset[:2])
'''
# cache_dir = './data/qa_mfa/'
#
# # 加载数据集，并指定缓存目录
# dataset = load_dataset(
#     "liwu/MNBVC",
#     'qa_mfa',
#     split='train',
#     # streaming=True,
#     cache_dir=cache_dir
# )
# print(dataset)
# # dataset["答案"]
# pprint(dataset[:10])
#
# cache_dir = './data/news_peoples_daily/'
#
#
# # 加载数据集，并指定缓存目录
# dataset = load_dataset(
#     "liwu/MNBVC",
#     'news_peoples_daily',
#     split='train',
#     # streaming=True,
#     cache_dir=cache_dir
# )
# print(dataset)
# # dataset["问"]
# # dataset["答"]
# pprint(dataset[:10])
#
#
#
# cache_dir = './data/new_york_times_news_2000_2007'
#
# # 加载数据集，并指定缓存目录
# dataset = load_dataset(
#     'ErikCikalleshi/new_york_times_news_2000_2007',
#     cache_dir=cache_dir,
#     split='train',
# )
# print(dataset)
# print(dataset[:2])
#
# cache_dir = './data/cnn'
#
# # 加载数据集，并指定缓存目录
# dataset = load_dataset(
#     'abisee/cnn_dailymail',
#     '3.0.0',
#     cache_dir=cache_dir,
#     split='train',
# )
# print(dataset)
# print(dataset[:2])