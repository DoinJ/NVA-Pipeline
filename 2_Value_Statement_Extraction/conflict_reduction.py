import argparse
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import networkx as nx
import numpy as np
import logging

# set up logging
logging.basicConfig(level=logging.INFO)

def semantic_similarity(stmt1, stmt2, llm):
    prompt = f"On a scale of 0 to 1, how semantically similar are these two statements?\n\nStatement 1: {stmt1}\nStatement 2: {stmt2}\n\nSimilarity score:"
    output = llm.generate(prompt_token_ids=tokenizer.encode(prompt), sampling_params=SamplingParams(temperature=0.2, max_tokens=1))[0]
    return float(tokenizer.decode(output.outputs[0].token_ids).strip())

def geospatial_distance(loc1, loc2):
    # implement geospatial distance calculation here
    return np.random.rand()  # placeholder

def social_network_similarity(org1, org2):
    # implement social network similarity calculation here  
    return np.random.rand()  # placeholder

def detect_conflicts(G, num_hops=5, similarity_threshold=0.5):
    cycles = nx.simple_cycles(G) 
    conflicting_edges = []
    
    for cycle in cycles:
        if len(cycle) == num_hops:
            stmts = [G.edges[cycle[i], cycle[(i+1) % num_hops]]['statement'] for i in range(num_hops)]
            
            if any(semantic_similarity(stmts[i], stmts[(i+1) % num_hops], llm) < similarity_threshold
                   for i in range(num_hops)):
                conflicting_edges.append(cycle[0], cycle[1])
    
    return conflicting_edges

def reduce_conflicts(G, num_iterations=5):
    for _ in range(num_iterations):
        conflicting_edges = detect_conflicts(G)
        G.remove_edges_from(conflicting_edges)
        
        # recalculate dominant value stance
        stances = nx.get_edge_attributes(G, 'stance')
        dominant_stance = pd.Series(stances).mode()[0]
        
        # remove edges that deviate from dominant stance
        for u, v in G.edges:
            if G.edges[u, v]['stance'] != dominant_stance:
                G.remove_edge(u, v)
        
    return G
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="gov_xuexiqiangguo",
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
    
    with open(f"data/{args.data_name}/filtered_statement_result.json") as f:
        data = json.load(f)
        
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    until = tokenizer.eos_token_id
    until = tokenizer.decode(until)
    until = [until, "<|end_of_text|>"]
    print(f"until: {until}")
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=2048,
        stop=until,
    )

    # Create an LLM.
    llm = LLM(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )

    G = nx.Graph()

    for article in data:
        G.add_node(article['raw_news'])
        
        for other_article in data:
            if article != other_article:
                G.add_edge(article['raw_news'], other_article['raw_news'], 
                           statement=article['statement'],
                           stance=article['filter_flag'],
                           sem_sim=semantic_similarity(article['statement'], other_article['statement'], llm)  
                           # geo_dist=geospatial_distance(article.get('location', ''), other_article.get('location', '')),
                           # soc_sim=social_network_similarity(article.get('org', ''), other_article.get('org', ''))
                           )
    
    G = reduce_conflicts(G)
    
    final_stmts = list(nx.get_edge_attributes(G, 'statement').values())
    
    with open(f"data/{args.data_name}/conflict_reduced_statements.json", 'w') as f:
        json.dump(final_stmts, f)