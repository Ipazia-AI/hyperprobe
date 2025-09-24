from os import path, makedirs
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import gc
import pandas as pd
import torch
import pickle
import zlib
from dotenv import load_dotenv

# LOCAL IMPORTS
from hyperprobe.data_creation.utils import emb_utils

def ingest_embeddings(docs:list[str], model_name:str, k_clusters:int = 5, input_as_analogy: bool= False, compute_eigenvalues = False, device = None) -> tuple[dict[str, torch.Tensor], tuple[int, int], pd.DataFrame, torch.Tensor]:
    """
    Ingest the embeddings from the LLM.
    """
        
    # (0) Create the dataset and loader
    inputs = emb_utils.InputDataset(docs)
    input_loader = DataLoader(inputs, batch_size = configs['batch_size'], shuffle = True)
    
    # (1a) Load the LLM
    tokenizer, model = emb_utils.load_llm(model_name, dtype = torch.bfloat16, device = device)
    median_layer = model.config.num_hidden_layers // 2
    probed_layers = (median_layer, model.config.num_hidden_layers)
    
    # (1c) Load the unembedding layer
    if hasattr(model, 'lm_head'):
        unembedding_layer = model.lm_head
    elif hasattr(model, 'embed_out'):
        unembedding_layer = model.embed_out
    else:
        raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
    
    # Get the tokens for the cutoffs
    #cutoff_tokens = np.array([tokenizer.encode(t, add_special_tokens=False)[0] for t in [' =', ' :']])
    
    # (1b) Create the embeddings
    token_embeddings = dict()
    feature_softmax = dict()
    cluster_stats = list()
    layer_correlations = list()
    eig_values = list()
    for input_batch in tqdm(input_loader, desc = 'Creating embeddings'):
        
        # Remove the target token
        if input_as_analogy:
            parts = input_batch[0].split()
            partial_doc = ' '+ ' '.join(parts[:-1])
            target_word = parts[-1].lower()
        else:
            partial_doc = input_batch[0]
            target_word = None
        
        # Tokenize the batch of prompts
        inputs_ids = tokenizer(partial_doc, return_tensors="pt").to(model.device)
        outputs = model(**inputs_ids, output_hidden_states=True)
                
        # Extract the hidden states
        hs = torch.stack(outputs.hidden_states).squeeze()
        hs = hs[median_layer:, -1]
        
        if hs.ndim != 2:
            print('SKIP: Invalid hidden states shape:', hs.shape, 'ndim:', hs.ndim)
            continue

        # Compute the correlations
        layer_correlations.append(torch.corrcoef(hs).detach().cpu().bfloat16())
        
        # Compute the redundancy
        if compute_eigenvalues:
            gram_matrix = hs @ hs.T
            gram_matrix = gram_matrix.detach().cpu().float().numpy()
            hs_eig_values = np.linalg.eigvalsh(gram_matrix)
            eig_values.append(hs_eig_values)
        
        # Cluster the hidden states by applying k-means
        cluster_assignments, centroids = emb_utils.kmeans_cuda(hs, K = k_clusters)
        
        # Save the cluster assignments
        cluster_assignments = list(zip(range(median_layer, model.config.num_hidden_layers + 1), cluster_assignments))
        cluster_stats.extend(cluster_assignments)
        
        # Store the embeddings
        token_embeddings[input_batch[0]] = centroids.detach().bfloat16().cpu()
        
    # Free the GPU memory and delete all variables
    del model, tokenizer, inputs_ids, outputs, hs, cluster_assignments, centroids, feature_softmax, unembedding_layer, input_loader, inputs, input_batch, partial_doc, target_word

    # Perform garbage collection and clear CUDA memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
   
    return token_embeddings, probed_layers, cluster_stats, layer_correlations, eig_values

configs = {
    'model_name': 'meta-llama/Llama-3.1-8B', #'EleutherAI/pythia-1.4b',#, 'EleutherAI/pythia-1.4b'
    'batch_size': 1
}

if __name__ == '__main__':
    
    load_dotenv()
    print(f'GPUs ({torch.cuda.device_count()}):\n' + '\n'.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    
    # Load the data --> texts | random_pairs | reversed_mixed_texts | features
    #input_data_path = path.join('data', 'splitted_data.json')
    #with open(input_data_path, mode = "r") as file:
    #    examples = json.load(file)
    #examples = [item for items in examples.values() for item in items] #np.random.choice(examples, size = 10000, replace = False).tolist()
    
    # Load the questions
    input_data_path = path.join('data', 'squad','squad_training.json')
    with open(input_data_path, mode = "r") as file:
        examples = [item['doc'] for item in json.load(file)]
        
    #examples = np.random.default_rng(seed = 102).choice(examples, size = 1000, replace=False).tolist()

    # Ingest the embeddings
    token_embeddings, probed_layers, cluster_stats, layer_correlations, eig_values = ingest_embeddings(
        docs = examples, model_name = configs['model_name'], compute_eigenvalues = False, 
        device = None)#torch.device('cuda:1'))

    # (1) Create the output folder
    dataset_name = path.basename(input_data_path).split('.')[0].split('_')
    main_folder_name = dataset_name[0].lower() + ''.join(word.capitalize() for word in dataset_name[1:])
    version_name = f'clustered_L{probed_layers[0]}_L{probed_layers[1]}'
    output_folder = path.join('outputs', 'embeddings', main_folder_name, version_name)
    makedirs(output_folder, exist_ok=True)
    
    # Save the embeddings
    print('\nSaving the embeddings...\n')
    with open(path.join(output_folder, 'embeddings.pkl.zlib'), mode = "wb") as file:
        file.write(zlib.compress(pickle.dumps(token_embeddings, protocol=pickle.HIGHEST_PROTOCOL), level = 9))
        
    # Stack the cluster stats
    cluster_stats = pd.DataFrame(cluster_stats, columns = ['layer', 'cluster'])
    correlations = torch.stack(layer_correlations)
    if len(eig_values) > 0:
        eig_values = np.stack(eig_values)#.mean(axis=0)        
            
    # Save the cluster stats
    emb_utils.plot_cluster(cluster_stats, output_folder)
    emb_utils.plot_correlations(correlations, output_folder, starting_layer_label=probed_layers[0])
    if len(eig_values) > 0:
        emb_utils.save_eigenvalues(eig_values, output_folder, starting_layer_label=probed_layers[0])
    
    # Print the results
    print(f'\nOK: Saved the embeddings for {len(examples)} inputs --> config: {configs}\n')