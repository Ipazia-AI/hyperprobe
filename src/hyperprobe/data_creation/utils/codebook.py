from collections import defaultdict
from os import makedirs, path
from matplotlib.ticker import PercentFormatter
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

def load_jsonFile(file_path):
    with open(file_path, mode ='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_unique_inputSemantics(inputs, max_semantic_depth):                                      
    labels = defaultdict(set)
    
    # Extract labels from the inputs and their depths
    for input in inputs:
        
        # Extract the lexical semantics for each feature
        for feature_values in input['features'].values():
            
            # Extract the attribute (lexical semantics)
            lexical_semantics = feature_values[-1]
            
            # Filter the lexical semantics
            if isinstance(lexical_semantics, list):
                
                # Check if the lexical semantics are verbs
                is_verb = any([item.endswith('-v') for item in lexical_semantics])
                
                # Add the lexical semantics to the labels
                # VERB: keep only the first element
                if is_verb or len(lexical_semantics) == 1:
                    labels[0].add(lexical_semantics[0])
                    
                # NOUN: Skip the first element and limit the depth
                else:
                    for pos, item in enumerate(lexical_semantics):
                        if pos != 0 and pos < max_semantic_depth:
                            labels[pos].add(item)

    # Convert the sets of labels to lists 
    labels = pd.DataFrame.from_dict(
        data = {item.lower(): pos for pos, items in labels.items() for item in items}, 
        orient='index', columns=['depth'],
        dtype='int8')
    
    # Sort the labels by their depth and delate the duplicates
    labels = labels.loc[labels.index.drop_duplicates()]
    labels = labels.sort_values(by='depth', ascending=False)
    return labels

def check_distribution(encodings, output_folder):
    full_sim = pairwise_cosine_similarity(encodings, zero_diagonal = True)
    
    stats = pd.Series(full_sim[np.triu_indices_from(full_sim, k=1)]).describe().round(2).to_dict()
    
    output_folder = path.join(output_folder, 'dist')
    makedirs(output_folder, exist_ok=True)
    
    # Histogram    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(full_sim[np.triu_indices_from(full_sim, k=1)], edgecolor='black', alpha=0.7, color='firebrick', bins=50)
    ax.set_title('Cosine Similarity Distribution')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Vectors')
    ax.grid(True)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(PercentFormatter())
    fig.tight_layout()
    fig.savefig(path.join(output_folder, 'hist.pdf'))
    
    # Heatmap    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(full_sim[np.triu_indices_from(full_sim, k=1)], vert=False, patch_artist=True, boxprops=dict(facecolor='firebrick', color='black'))
    ax.set_title('Cosine Similarity Distribution')
    ax.set_xlabel('Cosine Similarity')
    ax.grid(True)
    ax.xaxis.set_major_formatter(PercentFormatter())
    fig.savefig(path.join(output_folder, 'boxplot.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    return stats


def vocabulary_based_features():
    
    from transformers import AutoTokenizer
    from wn import Wordnet
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # Load the wordnet --> ERROR: sqlite3.OperationalError --> python -m wn download oewn:2023
    en = Wordnet('oewn:2023')  
    
    # Control the vocabulary
    vocabulary = set()
    for token in set(tokenizer.get_vocab().keys()):
        token = token.strip('Ġ').strip()
        
        # Check if the token is a word
        if len(token) > 2 and en.synsets(token):
            vocabulary.add(token.lower())
    return list(vocabulary)