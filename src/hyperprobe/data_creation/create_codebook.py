import json
from os import path, makedirs
import numpy as np
import torchhd
import pandas as pd
import torch

# LOCAL IMPORTS
import hyperprobe.data_creation.utils.codebook as codebook_utils

def create_codebook(concepts : list[str], vsa_dimension: int = 4096) -> pd.DataFrame:
    """
    Create a codebook for the hyperprobe package.
    
    Parameters
    ----------
    vsa_dimension : int
        The dimension of the VSA.
    seed : int
        The seed for the random generator.
    """
    
    # Data preparation (lowercase, unique concepts)
    concepts = np.unique([f.lower() for f in concepts])
    
    # Sort the items
    np.sort(concepts)
    
    # Random generator
    random_generator = torch.Generator().manual_seed(101)
    
    # Create the VSA encodings
    vsa_encodings = pd.DataFrame(
        data = torchhd.random(num_vectors = len(concepts), dimensions = vsa_dimension, vsa = 'MAP', generator = random_generator).numpy(),
        index = concepts)
    
    return vsa_encodings
    

if __name__ == '__main__':
    
    # Set the seed
    torch.manual_seed(101)

    # Load labelled data
    data = codebook_utils.load_jsonFile(file_path = path.join('data', 'features.json'))
    items = list(data.keys())
    #llm_vocabulary = codebook_utils.vocabulary_based_features()
    
    with open(path.join('data', 'squad', 'squad_dataset.json'), 'r') as file:
        squad_df = json.load(file)
    qa_items = set()
    for item in squad_df: 
        for f in item['question_features']:
            qa_items.add(f)
        for features in item['answer_features']:
            for f in features:
                qa_items.add(f)
    items = list(set(items).union(qa_items))
    
    # Load the function
    vsa_encodings = create_codebook(items, vsa_dimension = 4096)
    
    print(vsa_encodings)
    
    # Create the output folder
    output_folder = path.join('outputs', 'codebooks')
    makedirs(output_folder, exist_ok = True)
    
    # Save the VSA encodings
    vsa_encodings.to_parquet(path.join(output_folder, f'features.parquet'))
    
    # Compute cosine similarity
    stats = codebook_utils.check_distribution(torch.from_numpy(vsa_encodings.values), output_folder = output_folder)
    print('\nCODEBOOK:', len(vsa_encodings), 'items -->', stats, '\n')