from collections import Counter
from os import makedirs, path, getenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import torch
import numpy as np
import pandas as pd
import seaborn as sns

def load_llm(model_name: str, dtype: torch.dtype = torch.bfloat16, device:torch.device = None) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    
    # Load the Hugging Face token from the environment variable
    hf_token = getenv("HUGGINGFACEHUB_TOKEN")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = dtype, token = hf_token, device_map = {'': device} if device else 'auto')

    # Set the model to evaluation mode
    model.eval()
    
    # Visit the model to ensure it is on the correct device
    print(f"\nLoaded the LLM ({model_name}, {dtype}, {model.device} --> {torch.cuda.get_device_name(model.device)})\n")
    print('Model size:', round(model.get_memory_footprint() / 1024**3, 2), 'GB')
    
    return tokenizer, model

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        
        if isinstance(data, list):
            self.docs = data
        elif isinstance(list(data.values())[0], list):
            self.docs = [item for examples in data.values() for item in examples]
        elif isinstance(data, dict):
            self.docs = [example for example in data.keys()]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]

def kmeans_cuda(X, K, max_iters = 1000):

    # Convert to float if bfloat16
    original_dtype = X.dtype
    if original_dtype == torch.bfloat16:
        X = X.float()
        
    # Get the number of rows
    n_rows = X.shape[0]
    
    # Initialize centroids randomly from the data
    generator = torch.Generator(device = X.device).manual_seed(101)
    centroids = X[torch.randperm(n_rows, generator=generator, device=X.device)[:K]]

    for _ in range(max_iters):
        
        # Step 1: Compute pairwise distances between each row in X and the centroids
        distances = torch.cdist(X, centroids)

        # Step 2: Assign each point to the closest centroid
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Step 3: Recompute centroids (mean of assigned points)
        new_centroids = torch.stack([X[cluster_assignments == k].mean(dim=0) for k in range(K)])
        
        # Step 4: Check for convergence (if centroids do not change)
        if torch.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids

    # Re-convert to bfloat16 if input was bfloat16
    if original_dtype == torch.bfloat16:
        centroids = centroids.bfloat16()
        
    # Reindex the cluster assignments
    cluster_assignments = cluster_assignments.cpu().numpy()
    unique_labels, first_indices = np.unique(cluster_assignments, return_index=True)
    reindex_mapping = dict(zip(np.argsort(first_indices), unique_labels))
    cluster_assignments = [int(reindex_mapping[label]) for label in cluster_assignments]

    return cluster_assignments, centroids


def logit_lens(hs, projector, tokenizer):

    # Compute the logits and probabilities
    logits = projector(hs)
    probs = torch.nn.functional.softmax(logits, dim = -1) 

    # Get the top tokens
    top_scores, top_indices = probs.topk(k = 1)
    top_tokens = [(tokenizer.decode([index.item()]).strip('Ġ').strip(), prob.item()) for index, prob in zip(top_indices, top_scores)]
    
    # Get the top token
    top_token = top_tokens[0][0]
    token_softmax = top_tokens[0][1]

    return top_token, token_softmax

def plot_cluster(cluster_stats, output_folder):
    
    # Group the cluster stats by layer and normalize
    cluster_stats = cluster_stats.groupby('layer').agg(Counter).map(
        lambda counter: {key : value / sum(counter.values()) for key, value in counter.items()})
    cluster_stats = cluster_stats['cluster'].apply(pd.Series)
    
    # Create the output folder
    makedirs(output_folder, exist_ok=True)
    
    # Plot the cluster stats
    #sns.set_style('whitegrid')
    fig = plt.figure(figsize = (5, 5))
    ax = sns.heatmap(cluster_stats, cmap = 'Reds', annot = True, fmt = '.0%', 
                     cbar_kws={'shrink': 0.4, 'label': 'Inputs'}) # 'anchor': (-1.05, 0.7)} 
    
    ax.set_xlabel('Cluster', color ='firebrick', size = 12)
    ax.set_ylabel('Layer', color ='firebrick', size = 12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.grid(axis = 'x', color = 'black', linestyle = '--', linewidth = 0.2, alpha = 0.3)
    ax.grid(axis = 'y', color = 'black', linestyle = '--', linewidth = 0.2, alpha = 0.3)
    
    # Color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    cbar.ax.yaxis.set_label_position("left")

    plt.tight_layout()
    plt.savefig(path.join(output_folder, 'cluster_stats.pdf'))
    plt.close()
    
def plot_correlations(correlations, output_folder, starting_layer_label):

    # Create the output folder
    makedirs(output_folder, exist_ok=True)
    
    # Compute the average values
    avg_corr = torch.mean(correlations, dim = 0).float()
    
    # Plot the correlations
    fig = plt.figure(figsize = (10, 10))
    ax = sns.heatmap(avg_corr, cmap = 'coolwarm', annot = True, fmt = '.1g', # mask = np.triu(np.ones_like(avg_corr, dtype=bool)),
                     vmin = -1, vmax = 1, center = 0, cbar_kws={'shrink': 0.4, 'label': 'Correlation'}) 
    
    ax.set_xlabel('Layer', color ='firebrick', size = 12)
    ax.set_ylabel('Layer', color ='firebrick', size = 12)
    
    layer_labels = [str(starting_layer_label + i) for i in range(len(ax.get_yticklabels()))]
    ax.set_yticklabels(layer_labels, rotation=0)
    ax.set_xticklabels(layer_labels, rotation=0)
    
    # Color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.yaxis.set_label_position("left")
    
    fig.tight_layout()
    fig.savefig(path.join(output_folder, 'layer_corr.pdf'))
    plt.close()
    
    
def plot_silhouette_scores(scores, output_folder):
    
    # Create the output folder
    makedirs(output_folder, exist_ok=True)

    # Create the dataframe
    df = pd.DataFrame(scores)
    df = pd.melt(df, var_name = 'model', value_name = 'values')
    
    df = df.melt(id_vars='model', value_name='score_dict') \
       .drop('variable', axis=1) \
       .assign(score=lambda d: d['score_dict'].apply(lambda x: list(x.items()))) \
       .explode('score') \
       .assign(key=lambda d: d['score'].apply(lambda x: x[0]),
               value=lambda d: d['score'].apply(lambda x: x[1])) \
       .explode('value') \
       .drop(columns=['score_dict', 'score']) \
        .rename(columns={'key': 'cluster', 'value': 'score'}) \
    
    print(df)
    
    # Plot the silhouette scores
    fig = plt.figure(figsize = (9, 6))
    ax = sns.lineplot(data = df, y = 'score', x = 'cluster', hue = 'model', marker='o', color='firebrick', linewidth = 2, errorbar = 'sd', zorder=10) 
    
    # Annotate the chosen k
    ax.axhline(y=df['score'].mean(), color='black', linestyle='-.', alpha=0.7, lw = 1, label=f'Average score ({df["score"].mean().round(2)})', zorder = 2)
    
    plt.ylabel('Silhouette Score', color ='firebrick', size = 12)
    plt.xlabel('Number of Clusters (k)', color ='firebrick', size = 12)
    plt.xticks(df['cluster'].unique(), rotation = 0)
    plt.yticks(color = 'black')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.ylim(bottom = -0.05, top = 1)
    plt.grid(axis = 'x', color = 'black', linestyle = '-', linewidth = 0.2, alpha = 0.2)
    plt.grid(axis = 'y', color = 'black', linestyle = '--', linewidth = 0.2, alpha = 0.3)
    
    # Compute deltas
    model_stats = df[['cluster', 'score', 'model']].groupby(['cluster', 'model']).mean()
    model_stats = model_stats['score'].unstack()
    avg_std = model_stats.std(axis = 1).rename('model_std').to_frame().reset_index()
    std_value = avg_std['model_std'].mean().item()
    avg_std['plot'] = avg_std['model_std'] < std_value
    x_labels = avg_std[avg_std['plot']]['cluster'].values
    print(avg_std)
    
    split_indices = np.where(np.diff(x_labels) != 1)[0] + 1
    x_groups = np.split(x_labels, split_indices)

    for idk, group in enumerate(x_groups):
        print(group)
        ax.fill_betweenx(x1 = group[0], x2 = group[-1], y = [-0.05, 1], color='gray', alpha=0.2, zorder=1, label=f'Low variability across\nmodels (σ < {round(std_value, 2)})' if idk == 0 else None)
    
    ax.legend()
    fig.tight_layout()
    
    fig.savefig(path.join(output_folder, 'silhouette_scores.pdf'))
    plt.close()
    
    
def save_eigenvalues(eig_values, output_folder, starting_layer_label = 0):
    """
    Plot the eigenvalues of the Gram matrix.
    """
    
    # Create the output folder
    makedirs(output_folder, exist_ok=True)
    
    # Herfindahl-hirschman index (HHI)
    lambda_i = eig_values.mean(axis = 0) / np.sum(eig_values.mean(axis = 0))
    HHI = 1/sum(lambda_i**2)
    print('HHI:', round(HHI, 3))
    
    # Save the HHI to a JSON file
    pd.Series({'HHI': HHI}).to_json(path.join(output_folder, 'hhi.json'), index = False)
    
    #columns = pd.MultiIndex.from_tuples([
    #    ("Eigenvalue", "Mean (normalized)"),  
    #    ("Eigenvalue", "Mean"), 
    #    ("Eigenvalue", "Std")
    #])
    
    # Convert the eigenvalues to a DataFrame
    df = pd.DataFrame({
        'Eigenvalue (normalized mean)': eig_values.mean(axis = 0) / eig_values.mean(axis = 0).sum(),
        'Eigenvalue (mean)': eig_values.mean(axis = 0).round(2),
        'Eigenvalue (std)': eig_values.std(axis = 0).round(2)
    })
    print(df)
    
    # Save the DataFrame
    df.round(2).to_latex(path.join(output_folder, 'eigenvalues.tex'), index = True, escape = False)

    # Plot the eigenvalues with its standard deviation
    fig = plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x=df.index, y='Eigenvalue (mean)',  color='firebrick', capsize=0.1, err_kws={'color': 'gray'}, edgecolor='black', linewidth=1.5, zorder=10) 
    ax.set_xlabel("Component ", color ='firebrick', size = 12)
    ax.set_ylabel("Eigenvalue", color ='firebrick', size = 12)
    fig.tight_layout()
    fig.savefig(path.join(output_folder, 'eigenvalues.pdf'))
    plt.close()
    
    fig = plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x=df.index, y='Eigenvalue (normalized mean)',  color='firebrick', capsize=0.1, err_kws={'color': 'gray'}, edgecolor='black', linewidth=1.5, zorder=10) 
    ax.set_xlabel("Component ", color ='firebrick', size = 12)
    ax.set_ylabel("Normalized eigenvalue", color ='firebrick', size = 12)
    fig.tight_layout()
    fig.savefig(path.join(output_folder, 'norm_eigenvalues.pdf'))
    plt.close()
        
    