import logging
import os
import pickle

from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"
from sentence_transformers import SentenceTransformer
import BERTopic_utils
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech, TextGeneration
from bertopic import BERTopic
from torch import cuda
from torch import bfloat16
import transformers
import BERTopic_dataloader
import json
# GPU support
# pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
# pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
# pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
# pip install cupy-cuda12x -f https://pip.cupy.dev/aarch64
# Single GPU
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import argparse

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# CPU support
# from umap import UMAP
# from hdbscan import HDBSCAN

# Set the logging level
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # create a parser object
    parser = argparse.ArgumentParser(description="BERTopic main")

    # add arguments
    parser.add_argument("--sentence_model_name", type=str, default="BAAI/bge-small-zh-v1.5",
                        choices=["BAAI/bge-small-zh-v1.5", 
                                 "BAAI/bge-small-en-v1.5", 
                                 "BAAI/bge-multilingual-gemma2", 
                                 "dangvantuan/french-document-embedding", 
                                 "aari1995/German_Semantic_STS_V2",
                                 "dangvantuan/sentence-camembert-base",
                                 "antoinelouis/french-me5-small"])
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
    parser.add_argument("--min_cluster_size", type=int, default=50)

    args = parser.parse_args()

    dataset = BERTopic_dataloader.my_load_dataset(args.data_name)
    data_size = len(dataset)

    if args.data_name in [
        "pressreleases",
        "news_peoples_daily",
        "bbc",
        "cnn"
        "new_york_times",
        "french",
        "german"
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

    print(f"dataset: {dataset[:2]}")

    # Load the SentenceTransformer model
    sentence_model_name = args.sentence_model_name
    # embedding_model = SentenceTransformer(sentence_model_name)
    embedding_model = SentenceTransformer(sentence_model_name, trust_remote_code=True) # Set trust_remote_code=True for French and German embedding model
    # 如果有保存的SentenceTransformer对象，可以直接加载
    if os.path.exists(f"data/{args.data_name}_embedding/{sentence_model_name}.pkl"):
        logging.info(f"Loading data/{args.data_name}_embedding/ from file")
        embeddings = pickle.load(open(f"data/{args.data_name}_embedding/{sentence_model_name}.pkl", "rb"))
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

        if not os.path.exists(f"data/{args.data_name}_embedding/{sentence_model_name.split('/')[0]}"):
            os.makedirs(f"data/{args.data_name}_embedding/{sentence_model_name.split('/')[0]}")
        pickle.dump(embeddings, open(f"data/{args.data_name}_embedding/{sentence_model_name}.pkl", "wb"))
    logging.info(f"Embeddings computed. Shape: {embeddings.shape}")

    # Reduce dimensionality
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric='cosine',
        # metric='euclidean', # 由于上面正则化了，所以这里使用欧氏距离
        # output_metric="euclidean",
        random_state=42
    )

    # Cluster embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        # min_samples=data_size // 150,
        metric='euclidean',
        # metric='cosine',
        cluster_selection_method='eom',
        prediction_data=True
    )
    # 有一个参数可以控制主题的数量，即 nr_topics 。但是，此参数会在创建主题后合并主题。它是一个支持创建固定数量主题的参数。

    # Vectorize text & c-TF-IDF
    vectorizer_model = CountVectorizer(stop_words=stopword_lst, min_df=1, ngram_range=(1, 2))
    # vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    # 忽略停用词和不常用词。此外，通过增加 n 元语法范围，我们将考虑由一个或两个单词组成的主题表示。

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    # python -m spacy download en_core_web_sm
    # python -m spacy download zh_core_web_sm
    # pos_model = PartOfSpeech("zh_core_web_sm")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # Prompt
    prompt = """
    # <|start_header_id|>user<|end_header_id|>I have a topic that contains the following documents:
    # [DOCUMENTS]
    # The topic is described by the following keywords: [KEYWORDS]
    #
    # Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    # topic: <topic label>
    # <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # """

    model_id = args.llm_name
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Llama 2 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Llama 2 Model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=bfloat16,
    )
    model.eval()
    # model.to('cuda:1')

    # Our text generator
    generator = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.6,
        top_p=0.8,
        max_new_tokens=500,
        # repetition_penalty=1.1
    )

    # Text generation with Llama 2
    llama3 = TextGeneration(generator, prompt=prompt)
    #
    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "Llama3": llama3,
        "MMR": mmr_model,
        # "POS": pos_model
    }

    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=10,
        verbose=True
    )

    # Fit BERTopic
    logging.info("Fitting BERTopic......")
    topics, probs = topic_model.fit_transform(dataset, embeddings)

    # Save the model
    if not os.path.exists(f"data/{args.data_name}/my_model_dir"):
        os.makedirs(f"data/{args.data_name}/my_model_dir")
    topic_model.save(f"data/{args.data_name}/my_model_dir", serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=sentence_model_name)

    # logging topic_model.get_topic_info()
    logging.info("Topic information: \n{}".format(topic_model.get_topic_info()))
    # save the topic information
    topic_model.get_topic_info().to_csv(f"data/{args.data_name}/topic_info.csv")

    if not os.path.exists(f"data/{args.data_name}/visualization"):
        os.makedirs(f"data/{args.data_name}/visualization")

    # visualize the topics
    hierarchy_fig = topic_model.visualize_hierarchy(custom_labels=True)
    hierarchy_fig.write_html(f"data/{args.data_name}/visualization/visualize_hierarchy.html")

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    # We can also hide the annotation to have a more clear overview of the topics
    # You can hide the hover with `hide_document_hover=True` which is especially helpful if you have a large dataset
    documents_fig = topic_model.visualize_documents(dataset, reduced_embeddings=reduced_embeddings, custom_labels=True,
                                                    sample=0.01,
                                                    hide_annotations=True)
    documents_fig.write_html(f"data/{args.data_name}/visualization/visualize_documents.html")
    documents_fig.write_image(f"data/{args.data_name}/visualization/visualize_documents.png")

    topic_model.visualize_heatmap()
