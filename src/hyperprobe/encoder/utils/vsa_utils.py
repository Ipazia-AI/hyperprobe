import re
import numpy as np
import pandas as pd
import torchhd
import torch
import matplotlib.pyplot as plt
import warnings
from os import path, makedirs
from torchmetrics.functional import pairwise_cosine_similarity
from matplotlib.ticker import PercentFormatter

def create_vsa_encodings(item:dict, codebook:pd.DataFrame, codebook_set:set = None, verbose:bool=False) -> torch.Tensor:
    """Create the VSA encodings for the input sentences.

    Args:
        item (dict): input as a dictionary with: 'doc' (textual input), 'pair' (values of the target word).
        codebook (pd.DataFrame): VSA codebook with all VSA encodings
        codebook_set (set): List of all concepts in the codebook (for fast lookup).
        verbose (bool, optional): Verbosity. Defaults to True.

    Returns:
        torch.Tensor: VSA encoding of the input sentence.
    """
    
    # If the concept set is not provided, re-create it from the codebook index (inefficient).
    if codebook_set is None:
        codebook_set = set(codebook.index)

    # Split the document into tokens
    item['doc'] = item['doc'].strip()
    
    # STRAT 1: Split by '=' or ':'
    if '=' in item['doc'] and ':' in item['doc']:
        tokens = re.split(r"\s*[:=]\s*", item['doc'].strip())
    # STRAT 2: Split by whitespace
    else:
        tokens = item['doc'].split()
    
    # Remove empty tokens and convert to lowercase
    tokens = set(t.lower() for t in tokens if t)

    # Get the target pair and example pair as sets
    if len(tokens) == 1:
        target_pair, example_pair = set(), set()
    else:
        if isinstance(item['concepts'][0], str):
            target_pair = set(item.lower() for item in item['concepts'])
            example_pair = set(token for token in (tokens - target_pair))
        elif isinstance(item['concepts'][0], tuple):
            example_pair = set(item.lower() for item in item['concepts'][0])
            target_pair = set(item.lower() for item in item['concepts'][1])

    # Selected_concepts
    selected_concepts = tokens.copy()

    # Check if all selected concepts are present in the codebook, else skip document.
    if selected_concepts - codebook_set:
        warnings.warn(f"The following concepts are not in the codebook: {selected_concepts - codebook_set}", UserWarning)
        
        # Remove the missing concepts from the selected concepts
        selected_concepts = selected_concepts.intersection(codebook_set)

    concepts = []
    remaining_concepts = selected_concepts.copy()
    
    # If all example_pair items are in the remaining concepts, get their encoding and remove them.
    vsa_operations = []
    if example_pair and remaining_concepts and example_pair.issubset(remaining_concepts):
        
        # Bind the key and value of the pair
        example_encoding = torchhd.multibind(torchhd.MAPTensor(codebook.loc[list(example_pair)].values).to(torch.int8))
        
        # Save the results
        concepts.append(example_encoding)
        remaining_concepts -= example_pair
        vsa_operations.append('(' +' ⊙ '.join(example_pair) + ')')

    # If all target_pair items are in the selected concepts, get their encoding and remove them.
    if target_pair and remaining_concepts and target_pair.issubset(remaining_concepts):
        
        # Bind the key and value of the pair
        target_encoding = torchhd.multibind(torchhd.MAPTensor(codebook.loc[list(target_pair)].values).to(torch.int8))
        
        # Save the results
        concepts.append(target_encoding)
        remaining_concepts -= target_pair
        vsa_operations.append('(' +' ⊙ '.join(target_pair) + ')')

    # Retrieve the item encodings from the codebook for any remaining concepts.
    single_encodings = None
    if len(remaining_concepts) > 0:
        single_encodings = torchhd.MAPTensor(codebook.loc[list(remaining_concepts)].values).to(torch.int8)
        
        if verbose:
            print(f"[{item['doc']}] single_encodings:", remaining_concepts, ' -->', single_encodings)

    # Combine encodings based on the number of concept groups
    if len(concepts) == 2:
        vsa = torchhd.multiset(torch.stack(concepts)).normalize()
        vsa_operations = ' + '.join(vsa_operations)
    elif len(concepts) == 1:
        if single_encodings is not None:
            vsa = torchhd.multiset(torch.stack([concepts[0], single_encodings])).normalize()
        else:
            vsa = concepts[0]
    else:
        vsa = torchhd.multiset(single_encodings).normalize()

    # Save the encoding of the document
    vsa = vsa.as_subclass(torch.Tensor).to(torch.int8)
    
    if verbose:
        print(f"\nDOC: '{item['doc'].strip()}'")
        print('--> TOKENS:', tokens)
        print('--> PAIRS:', 'TARGET:', target_pair, '| IN-CONTEXT:', example_pair)
        print('--> VSA OPERATIONS:', vsa_operations)
        print('--> VSA ENCODING:', vsa.size(), '-->', vsa)
    
    return vsa

def check_distribution(vsa_encodings, output_folder):
    
    # Get the VSA encodings
    encodings = torch.stack([item['vsa'].float() for item in vsa_encodings])
    print(f'\n[INFO] Checking the distribution of the VSA encodings (size: {list(encodings.shape)})...')
    
    # Compute the cosine similarity between the encodings
    full_sim = pairwise_cosine_similarity(encodings, zero_diagonal = True)

    # Create the output folder
    output_folder = path.join(output_folder, 'vsaDistribution')
    makedirs(output_folder, exist_ok=True)
    
    # Boxplot for cosine similarity
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(full_sim[np.triu_indices_from(full_sim, k=1)], vert=False, patch_artist=True, boxprops=dict(facecolor='firebrick', color='black'))
    ax.set_title('Cosine Similarity Distribution')
    ax.set_xlabel('Cosine Similarity')
    ax.grid(True)
    ax.xaxis.set_major_formatter(PercentFormatter())
    fig.savefig(path.join(output_folder, 'boxplot.png'), bbox_inches='tight')
    plt.close(fig)
    
    # Histogram for cosine similarity
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(full_sim[np.triu_indices_from(full_sim, k=1)], edgecolor='black', alpha=0.7, color='firebrick', bins=50)
    ax.set_title('Cosine Similarity Distribution')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Vectors')
    ax.grid(True)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(PercentFormatter())
    fig.tight_layout()
    fig.savefig(path.join(output_folder, 'hist.png'))
    plt.close(fig)
