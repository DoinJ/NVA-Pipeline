# Define embedding model
import json
import logging
import os
import pickle

import BERTopic_utils
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import BERTopic_dataloader

dataset = BERTopic_dataloader.load_dataset()
# Get stop words
if not os.path.exists("data/stopwords.json"):
    stopword_lst = BERTopic_utils.get_stop_words(dataset)
    with open("data/stopwords.json", "w", encoding="utf-8") as f:
        json.dump(stopword_lst, f, ensure_ascii=False, indent=4)
else:
    with open("data/stopwords.json", "r", encoding="utf-8") as f:
        stopword_lst = json.load(f)
# drop the stop words in the training process, but keep the stop words in the inference process
dataset = BERTopic_utils.drop_stop_words(dataset, stopword_lst)

sentence_model_name = "BAAI/bge-small-zh-v1.5"
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# Load model and add embedding model
loaded_model = BERTopic.load("my_model_dir", embedding_model=embedding_model)


# 如果有保存的SentenceTransformer对象，可以直接加载
if os.path.exists(f"data/appledaily_embedding/{sentence_model_name}.pkl"):
    logging.info("Loading embeddings from file")
    embeddings = pickle.load(open(f"data/appledaily_embedding/{sentence_model_name}.pkl", "rb"))
else:
    # Pre-calculate embeddings
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info(f"Calculating embeddings from scratch using {sentence_model_name}")
    # embeddings = embedding_model.encode(dataset, show_progress_bar=True, normalize_embeddings=True)
    # Start the multi-process pool on all available CUDA devices
    pool = embedding_model.start_multi_process_pool()
    # Compute the embeddings using the multi-process pool
    embeddings = embedding_model.encode_multi_process(dataset, pool, batch_size=64, normalize_embeddings=True)
    # Optional: Stop the processes in the pool
    embedding_model.stop_multi_process_pool(pool)

    if not os.path.exists(f"data/appledaily_embedding/{sentence_model_name.split('/')[0]}"):
        os.makedirs(f"data/appledaily_embedding/{sentence_model_name.split('/')[0]}")
    pickle.dump(embeddings, open("data/appledaily_embedding/{}.pkl".format(sentence_model_name), "wb"))
logging.info(f"Embeddings computed. Shape: {embeddings.shape}")

# visualize the topics
topic_model = loaded_model
hierarchy_fig = topic_model.visualize_hierarchy(custom_labels=True)
hierarchy_fig.write_html("data/visualization/visualize_hierarchy.html")

from cuml.manifold import UMAP
# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
# We can also hide the annotation to have a more clear overview of the topics
# You can hide the hover with `hide_document_hover=True` which is especially helpful if you have a large dataset
documents_fig = topic_model.visualize_documents(dataset, reduced_embeddings=reduced_embeddings, custom_labels=True,
                                                sample=0.001,
                                                hide_annotations=True)
documents_fig.write_html("data/visualization/visualize_documents.html")
documents_fig.write_image("data/visualization/visualize_documents.png")

topic_model.visualize_heatmap()


