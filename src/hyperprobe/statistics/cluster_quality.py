from os import path, makedirs
import pickle
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import torch

# LOCAL IMPORTS
from hyperprobe.data_creation.utils import emb_utils
    
if __name__ == '__main__':
    
    print(f'GPUs ({torch.cuda.device_count()}):\n' + '\n'.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    
    # Load the codebook
    codebook = pd.read_parquet(path.join('outputs', 'codebooks', 'features.parquet'))
    
    # Load the data --> texts | random_pairs | reversed_mixed_texts | features
    input_data_path = path.join('data', 'splitted_data.json')
    with open(input_data_path, mode = "r") as file:
        examples = json.load(file)
    examples = [item for items in examples.values() for item in items]
    
    # Debug
    examples = np.random.choice(examples, size = 10000, replace = False).tolist()   
    
    # (0) Load the inputs
    inputs = emb_utils.InputDataset(examples)
    input_loader = DataLoader(inputs, batch_size = 1, shuffle = True)
    
    models = ['meta-llama/Llama-4-Scout-17B-16E', "allenai/OLMo-2-0325-32B", "microsoft/phi-4", "meta-llama/Llama-3.1-8B", 'EleutherAI/pythia-1.4b', 'openai-community/gpt2-medium']
    model_name = models[-1]
    cluster_scores = defaultdict(list)

    # (1a) Load the LLM
    tokenizer, model = emb_utils.load_llm(model_name, dtype = torch.bfloat16)
    median_layer = model.config.num_hidden_layers // 2
    version_name = f'clustered_L{median_layer}_L{model.config.num_hidden_layers}'
    
    # (1b) Load the unembedding layer
    if hasattr(model, 'lm_head'):
        unembedding_layer = model.lm_head
    elif hasattr(model, 'embed_out'):
        unembedding_layer = model.embed_out
    else:
        raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
    
    # Get the tokens for the cutoffs
    cutoff_tokens = np.array([tokenizer.encode(t, add_special_tokens=False)[0] for t in [' =', ' :']])
    
    # (1b) Create the embeddings
    cluster_stats = list()
    layer_correlations = list()
    silhouette_scores = defaultdict(list)
    for input_batch in tqdm(input_loader, desc = 'Creating embeddings'):
        
        # Remove the target token
        parts = input_batch[0].split()
        partial_doc = ' '+ ' '.join(parts[:-1])
        target_word = parts[-1].lower()
        
        # Tokenize the batch of prompts
        inputs_ids = tokenizer(partial_doc, return_tensors="pt").to(model.device)
        outputs = model(**inputs_ids, output_hidden_states=True)

        # Get the position of the last delimiter
        intra_delimiter_pos = torch.argwhere(inputs_ids.input_ids[0] == cutoff_tokens[1]).squeeze().tolist()
        if isinstance(intra_delimiter_pos, list):
            if len(intra_delimiter_pos) > 1:
                intra_delimiter_pos = intra_delimiter_pos[-1]
            else:
                intra_delimiter_pos = -1
                
        # Extract the hidden states
        hs = torch.stack(outputs.hidden_states).squeeze()
        hs = hs[median_layer:, intra_delimiter_pos]
        
        # Compute the correlations
        layer_correlations.append(torch.corrcoef(hs).detach().cpu().bfloat16())
        
        # Cluster the hidden states by applying k-means
        for k in range(2, hs.size(0)):
            cluster_assignments, centroids = emb_utils.kmeans_cuda(hs, K = k)
            sil_coeff = silhouette_score(X = hs.cpu().detach().float().numpy(), labels = cluster_assignments)
            silhouette_scores[k].append(sil_coeff.item())
        
        # Save the cluster assignments
        cluster_assignments = list(zip(range(median_layer, model.config.num_hidden_layers + 1), cluster_assignments))
        cluster_stats.extend(cluster_assignments)
    

    # Stack the cluster stats
    cluster_stats = pd.DataFrame(cluster_stats, columns = ['layer', 'cluster'])
    correlations = torch.stack(layer_correlations)
    
    # Save the cluster scores
    model_name = model_name.split('/')[-1]
    cluster_scores[model_name].append(silhouette_scores)    

    # Create the output folder
    output_folder = path.join('outputs', 'embeddings', 'cluster_stats')
    model_folder = path.join(output_folder, model_name)
    makedirs(model_folder, exist_ok=True)
            
    # Save the cluster stats
    emb_utils.plot_cluster(cluster_stats, model_folder)
    emb_utils.plot_correlations(correlations, model_folder, starting_layer_label = median_layer)
    
    # Load the previous cluster scores if they exist
    cluster_scores_path = path.join(output_folder, 'cluster_scores.pkl')
    if path.exists(cluster_scores_path):
        with open(cluster_scores_path, mode = "rb") as file:
            previous_cluster_scores = pickle.loads(file.read())
        
        # Merge the new cluster scores with the previous ones
        cluster_scores = previous_cluster_scores | cluster_scores
        
    # Save the cluster scores by appending to the file
    with open(cluster_scores_path, mode = "wb") as file:
        file.write(pickle.dumps(cluster_scores, protocol=pickle.HIGHEST_PROTOCOL))
    
    # Create the graph
    emb_utils.plot_silhouette_scores(cluster_scores, output_folder)
    
    # Print the results
    print(f'\nDONE ({len(inputs)} inputs)\n')
    
#from os import path
#import pickle
#from hyperprobe.data_creation.utils import emb_utils

#output_folder = path.join('outputs', 'embeddings', 'cluster_stats')
#cluster_scores_path = path.join(output_folder, 'cluster_scores.pkl')
#if path.exists(cluster_scores_path):
#    with open(cluster_scores_path, mode = "rb") as file:
#        cluster_scores = pickle.loads(file.read())
        
#emb_utils.plot_silhouette_scores(cluster_scores, output_folder)