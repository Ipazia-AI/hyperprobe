from collections import defaultdict
import json
from os import path
import numpy as np
import pandas as pd
from os import makedirs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

def extract_values(df):
    
    # Compute the length of the input and target
    df['target_length'] = df['target'].apply(lambda items: len(items[0]))
   
    # Extract the values of the vsa similarities
    vsa_sim = df['vsa_sim_before'].apply(lambda obj: list(obj.values()))
    df['vsa_sim_before_items'] = vsa_sim.apply(len)
    df['vsa_sim_before_avg'] = vsa_sim.apply(lambda x: round(np.mean(x), 4) if len(x) > 0 else 0)
    df['vsa_sim_before_std'] = vsa_sim.apply(lambda x: round(np.std(x), 4) if len(x) > 0 else 0)
    
    vsa_sim = df['vsa_sim_after'].apply(lambda obj: list(obj.values()))
    df['vsa_sim_after_items'] = vsa_sim.apply(len)
    df['vsa_sim_after_avg'] = vsa_sim.apply(lambda x: round(np.mean(x), 4) if len(x) > 0 else 0)
    df['vsa_sim_after_std'] = vsa_sim.apply(lambda x: round(np.std(x), 4) if len(x) > 0 else 0)

    # Extract the values from the nested dictionaries/lists
    extracted_factors = df.apply(
        lambda df_row: {k: len(v.split('|'))  for k, v in df_row['split_extracted_factors'].items()}, axis = 1).apply(pd.Series).add_prefix('extracted_before_').fillna(0)
    extracted_factors = extracted_factors.div(extracted_factors.sum(axis=1), axis=0)
    extracted_factors_after = df.apply(
        lambda df_row: {k: len(v.split('|'))  for k, v in df_row['split_extracted_factors_after'].items()}, axis = 1)\
            .apply(pd.Series).add_prefix('extracted_after_').fillna(0)
    extracted_factors_after = extracted_factors_after.div(extracted_factors_after.sum(axis=1), axis=0)
    
    llm_output_eval = df['llm_output_eval'].apply(pd.Series).add_prefix('llm_')
    jaccard = df['jaccard_scores'].apply(pd.Series).add_prefix('jaccard_')
    fuzzyjaccard = df['fuzzy_jaccard_scores'].apply(pd.Series).add_prefix('fuzzyjaccard_')
    semantic_sim = df['semantic_similarity_scores'].apply(pd.Series).add_prefix('semantic_')
    
    # Combine all the data
    stats = pd.concat([df, llm_output_eval, extracted_factors, extracted_factors_after, semantic_sim, jaccard, fuzzyjaccard], axis=1)
    
    # Compute additional metrics
    stats['llm_mentioned_in_answer'] = stats['llm_mentioned_in_answer'].apply(int)
    for label in ['question', 'answer', 'other']:
        stats['extracted_afterBefore_' + label + '_diff'] = stats['extracted_after_' + label] - stats['extracted_before_' + label]
        stats['extracted_afterBefore_' + label + '_ratio'] = stats['extracted_after_' + label] / (stats['extracted_before_' + label] + 1e-6)
    stats['semantic_answerQuestion_before_ratio'] = stats['semantic_extracted_before_answer'] / (stats['semantic_extracted_before_question'] + 1e-6)
    stats['semantic_answerQuestion_after_ratio'] = stats['semantic_extracted_after_answer'] / (stats['semantic_extracted_after_question'] + 1e-6)
    
    # Drop the original columns
    stats = stats.drop(columns=['vsa_sim', 'llm_output_eval', 'jaccard_scores', 'fuzzy_jaccard_scores', 'semantic_similarity_scores']) # 'vsa_sim_after',
    
    return stats

def create_boxplot(df, title = None):

    # Melt the dataframe to long format
    cols = ['semantic_extracted_before_question', 'semantic_extracted_after_question', 'semantic_extracted_before_answer', 'semantic_extracted_after_answer'] #'semantic_before_after_overlap', 'semantic_before_after_overlap']
    
    to_plot_all = df[cols].melt(var_name='Metric', value_name='Value')
    to_plot_wrong = df.loc[df['llm_mentioned_in_answer'] == False][cols].melt(var_name='Metric', value_name='Value')
    to_plot_wrong['Metric'] = "[WRONG] " + to_plot_wrong['Metric']
    to_plot = pd.concat([to_plot_all, to_plot_wrong], axis=0)
    to_plot['original_metric'] = to_plot['Metric'].str.lstrip('[WRONG] ').apply(lambda x: cols.index(x))
    to_plot['is_wrong'] = to_plot['Metric'].str.contains('WRONG')
    to_plot.loc[to_plot['is_wrong'] == False, 'Metric'] = '[ALL] ' + to_plot.loc[to_plot['is_wrong'] == False, 'Metric']
    to_plot = to_plot.sort_values(by=['original_metric', 'is_wrong'], ascending=True).drop(columns=['original_metric', 'is_wrong'])
    to_plot['Metric'] = to_plot['Metric'].str.replace('semantic_', '').str.replace('extracted_', '').str.replace('_', ' ')
    
    # COMPUTE DIFFERENCES
    num_diff = {}
    for col in cols: 
        all_mean = df[col].mean()
        wrong_mean = df.loc[df['llm_mentioned_in_answer'] == False, col].mean()
        diff = wrong_mean - all_mean
        num_diff[col] = round(diff, 4)
        print(f'Difference in means ({col}): {diff:.4f} (ALL: {all_mean:.4f}, WRONG: {wrong_mean:.4f})')
        #print('Differences in means (WRONG - ALL):\n', num_diff)
    
    # Define colors for each category

    # Create the boxplot
    fig, axes = plt.subplots(ncols = 2, figsize=(10, 2))
    
    # Plot each category in a separate subplot
    for idk, label in enumerate(['question', 'answer']):
        tmp = to_plot[to_plot['Metric'].str.contains(label)].copy()
        tmp['Metric'] = tmp['Metric'].str.replace(label, '').str.strip()

        # Define a color palette
        colors = {label: '#B45253' if 'WRONG' in label else '#84994F' for label in tmp['Metric'].unique()}
        sns.boxplot(data=tmp , y='Metric', x='Value', palette = colors, hue = 'Metric', legend=False, ax = axes[idk]) 

        # Customize the plot
        axes[idk].set_ylabel("")
        axes[idk].grid(axis='x', linestyle='--', alpha=0.6)
        axes[idk].set_xlabel("Semantic similarity score")
        axes[idk].set_title('FEATURES OF THE ' + label.upper() if label != 'overlap' else 'OVERLAPPING FEATURES')
    
    num_errors = len(df.loc[df['llm_mentioned_in_answer'] == False]) / len(df)
    fig.legend(['All data', f"LLM error set ({num_errors:.0%})"], loc='upper center', ncol=1, bbox_to_anchor=(0.6, 0.27))

    fig.tight_layout()

    
    return fig

def compute_correlations(df, method = 'pearson'):
    
    # Select only numeric columns
    cols = df.select_dtypes(exclude='object').columns.tolist()
    n_cols = len(cols)
    
    # Select the correlation method
    if method == 'pearson':
        corr_func = pearsonr
    elif method == 'spearman':
        corr_func = spearmanr
    elif method == 'kendall':
        corr_func = kendalltau
    
    # Initialize matrices
    corr_matrix = np.zeros((n_cols, n_cols))
    pval_matrix = np.zeros((n_cols, n_cols))
    
    # Compute correlations
    for i in range(n_cols):
        for j in range(n_cols):
            if np.std(df[cols[i]]) == 0 or np.std(df[cols[j]]) == 0:
                corr = 0.0 
                pval = 1.0
            else:
                corr, pval = corr_func(df[cols[i]], df[cols[j]])
            corr_matrix[i, j] = corr
            pval_matrix[i, j] = pval
            
    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols).round(2)
    pval_df = pd.DataFrame(pval_matrix, index=cols, columns=cols)
    
    return corr_df, pval_df
 

if __name__ == '__main__':
        
    # Load the questions
    version = 'QA_llama3_QA_bundle_equal2'
    root_folder = path.join('outputs', 'probing')
    file_path = path.join(root_folder, f'{version}.json')
    df = pd.read_json(file_path)
    
    # Create the output folder
    output_folder = path.join(root_folder, 'stats')
    makedirs(output_folder, exist_ok = True)
    
    # Extract the values
    df = extract_values(df)
    df.to_excel(path.join(output_folder, path.basename(file_path).strip('.json') + f'_all_data.xlsx'), index = False)
    
    # Create the boxplots
    fig = create_boxplot(df)
    fig.savefig(path.join(output_folder, f'boxplot.pdf'), dpi = 400)
    plt.close(fig)
    
    # Compute the metrics
    for label in df['llm_mentioned_in_answer'].unique().tolist() + ['all']:
        
        partial_df = df if label == 'all' else df[df['llm_mentioned_in_answer'] == label].copy()

        # Compute the stats
        stats = partial_df.describe()

        # Save the stats
        stats.to_excel(path.join(output_folder, path.basename(file_path).strip('.json') + f'_stats_{bool(label) if label != "all" else "All"}.xlsx'))
        
        graph_folder = path.join(output_folder, 'graphs')
        boxplot_folder = path.join(graph_folder, 'boxplots')
        corr_folder = path.join(graph_folder, 'correlations')
        makedirs(boxplot_folder, exist_ok = True)
        makedirs(corr_folder, exist_ok = True)
        
        # Compute the correlations
        partial_df.columns = [col.replace('_', ' ') for col in partial_df.columns]
        for method in ['pearson', 'spearman']: #  'kendall'

            # Compute the correlations and p-values
            cols = ['target length', 'vsa sim before items', 'llm f1', 'llm em', 'llm mentioned in answer', 
                    'extracted before question', 'extracted after question', 'extracted before answer', 'extracted after answer', 'extracted before other', 'extracted after other', 'semantic extracted before question', 'semantic extracted after question', 
                    'semantic extracted before answer', 'semantic extracted after answer', 'semantic question answer overlap', 'semantic before after overlap']
            corr, pvalues = compute_correlations(partial_df[cols].select_dtypes(exclude='object'), method = method)
            pvalues.to_excel(path.join(corr_folder, f'pvalues_{method}_{bool(label) if label != "all" else "All"}.xlsx'))
            
            # Extract significant correlations
            signficant_correlations = defaultdict(dict)
            for _, row in corr.iterrows():
                for col, value in row.items():
                    if value >= 0.2 and row.name != col:
                        signficant_correlations[row.name][col] = round(value, 2)
                    
            # Sort the dictionary by values in descending order
            signficant_correlations = {k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True)) for k, v in signficant_correlations.items()}
            
            with open(path.join(corr_folder, f'significant_corr_{method}_{bool(label) if label != "all" else "All"}.json'), 'w') as f:
                json.dump(signficant_correlations, f, indent=4)       
            
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(corr, cmap='coolwarm', center=0, vmax=1, vmin= -1, linewidths=0.5, linecolor="lightgray",
                        annot=True, fmt='.1f', square=True, mask = np.eye(corr.shape[0], dtype=bool),
                        cbar_kws={'label': 'Correlation Coefficient','shrink': 0.4, 'pad': 0.15,
                                'format': '%.1f', 'ticks': [-1, -0.5, 0, 0.5, 1]}, ax = ax)
            
            # Move the colorbar to the left
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_ticks_position('left')
            
            # Set the title
            ax.set_title(f'{method.capitalize()} Correlation')
            
            fig.tight_layout()
            fig.savefig(path.join(corr_folder, f'corr_{method}_{bool(label) if label != "all" else "All"}.pdf'))
            plt.close(fig)
            
            # Plot the p-values on a heatmap
            median = np.median(pvalues.values[np.triu_indices_from(pvalues.values, k=1)])
            print(f'Median p-value ({method}, {label}): {median:.2e}')
            fig, ax = plt.subplots(figsize=(16, 14))
            sns.heatmap(pvalues, cmap='coolwarm', annot=True, fmt='.0e', square=True, linewidths=0.5, linecolor="lightgray", vmax = 0.05, vmin =1e-99, #center = median, # vmax = 1, vmin = 0, 
                        mask = np.eye(pvalues.shape[0], dtype=bool),
                        cbar_kws={'label': 'P-value', 'shrink': 0.4, 'pad': 0.15 }, ax = ax) # 'format': '%.1e', 'ticks': [0, 1e-100, 1e-50, 0.05, 0.5, 1]
            fig.tight_layout()
            fig.savefig(path.join(corr_folder, f'pvalues_{method}_{bool(label) if label != "all" else "All"}.pdf'))
            