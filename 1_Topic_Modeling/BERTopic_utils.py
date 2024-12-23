import logging
import re
import time
from multiprocessing import Pool

from tqdm import tqdm

# set the logging level
logging.basicConfig(level=logging.INFO)
import nltk
from nltk.corpus import stopwords
import zhconv
from collections import Counter


def get_stop_words(dataset_lst):
    """
    Get the stop words
    :return: list[str]
    """
    # 下载停用词列表
    logging.info("Geting stopwords")
    nltk.download('stopwords')
    # 获取中文停用词列表
    chinese_stop_words = list(stopwords.words('chinese'))
    traditional_chinese_stop_words = [zhconv.convert(word, 'zh-hk') for word in chinese_stop_words]
    english_stop_words = list(stopwords.words('english'))
    french_stop_words = list(stopwords.words('french'))
    german_stop_words = list(stopwords.words('german'))

    # 从dataset_lst将词频最高的前100个词作为停用词
    # 分词
    word_counts = Counter()
    splited_dataset = []
    sub_dataset = dataset_lst[:50000]
    for doc in tqdm(sub_dataset, desc="分词", total=len(sub_dataset)):
        words = nltk.word_tokenize(doc)
        splited_dataset.append(words)
        # 统计词频
        word_counts += Counter(words)
    # 打印词频最高的前 10 个词
    most_common_words = word_counts.most_common(100)
    logging.info(f"词频最高的前 100 个词:{most_common_words}")
    # 将词频最高的词作为停用词
    my_stopwords = [word for word, count in most_common_words]

    stopword_lst = [*chinese_stop_words, *traditional_chinese_stop_words, *english_stop_words, *french_stop_words, *german_stop_words, *my_stopwords]

    logging.info(f"Stopwords generated!")
    return stopword_lst

def drop_stop_words_helper(args):
    doc, stopword_lst = args
    # 创建正则表达式模式，匹配所有停用词
    stopword_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, stopword_lst)) + r')\b')

    # 使用正则表达式替换停用词为空字符串
    filtered_doc = stopword_pattern.sub('', doc)
    return filtered_doc

def drop_stop_words(dataset, stopword_lst):
    """
    Drop the stop words from the dataset
    :param dataset: list[str]
    :param stopword_lst: list[str]
    :return: list[str]
    """
    logging.info("Dropping stopwords")
    start_time = time.time()
    with Pool(64) as pool:
        arg_lst = [(doc, stopword_lst) for doc in dataset]
        new_dataset = list(pool.map(drop_stop_words_helper, arg_lst))
    logging.info(f"Drop Stop Word Time taken: {time.time() - start_time}")

    return new_dataset
