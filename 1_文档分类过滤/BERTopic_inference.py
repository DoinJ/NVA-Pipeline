import json
import logging
import os
import argparse

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic

import BERTopic_dataloader
import BERTopic_utils

# set the logging level
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # create a parser object
    parser = argparse.ArgumentParser(description="BERTopic main")

    # add arguments
    parser.add_argument("--sentence_model_name", type=str, default="BAAI/bge-small-zh-v1.5",
                        choices=["BAAI/bge-small-zh-v1.5", "BAAI/bge-small-en-v1.5"])
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

    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Define embedding model
    embedding_model = SentenceTransformer(args.sentence_model_name)

    # Load model and add embedding model
    loaded_model = BERTopic.load(f"data/{args.data_name}/my_model_dir", embedding_model=embedding_model)

    df = loaded_model.get_topic_info()
    # df.to_csv("data/topic_info.csv")

    # browser the documents
    dataset = BERTopic_dataloader.my_load_dataset(args.data_name)
    data_size = len(dataset)

    if args.data_name in [
        "appledaily", "pressreleases",
        "news_peoples_daily",
        "bbc",
        "cnn",
        "new_york_times",
    ]:
        # Get stop words
        # create dir
        if not os.path.exists(f"data/{args.data_name}"):
            os.makedirs(f"data/{args.data_name}")
        if not os.path.exists(f"data/{args.data_name}/stopwords.json"):
            stopword_lst = BERTopic_utils.get_stop_words(dataset)
            with open(f"data/{args.data_name}/stopwords.json", "w", encoding="utf-8") as f:
                json.dump(stopword_lst, f, ensure_ascii=False, indent=4)
        else:
            with open(f"data/{args.data_name}/stopwords.json", "r", encoding="utf-8") as f:
                stopword_lst = json.load(f)
        # drop the stop words in the training process, but keep the stop words in the inference process
        dataset = BERTopic_utils.drop_stop_words(dataset, stopword_lst)

    logging.info("Getting document info")
    document_info_df = loaded_model.get_document_info(docs=dataset)
    logging.info("Document done!")
    # document_info_df = document_info_df[["Document", "Topic"]]
    document_info_df.to_csv(f"data/{args.data_name}/document_info.csv", index=False, encoding="utf-8")
    logging.info("Document info saved!")
