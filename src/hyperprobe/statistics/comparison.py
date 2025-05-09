from collections import defaultdict
from os import path, listdir
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator

def create_graph(scores):

    # Data preparation
    df = scores[['next_token_precision@1', 'next_token_precision@5', 'precision@1', 'precision@5', 'LLM']]
    df = df.sort_values(by=['next_token_precision@1'])

    # Melt the df
    df = pd.melt(df, id_vars='LLM', var_name='metric', value_name='value')
    
    # Create the plot
    plt.figure(figsize=(11, 5))
    sns.set_theme(style="whitegrid")
    plt.rc('font',**{'family':'sans-serif','sans-serif':['dejavu sans']})
    
    # Barplot
    ax = sns.barplot(data = df, x='metric', y='value', hue = 'LLM', palette='Set2', dodge=True, zorder = 1)
    
    # Add the average values
    avg_llm = scores[['next_token_precision@1']].values.flatten().mean() #  'next_token_precision@5'
    avg_probing = scores[['precision@1']].values.flatten().mean() # 'precision@5'
    ax.axhline(y=avg_llm, xmax = 0.49, color='black', linestyle=(5, (10, 3)), alpha = 0.5, zorder = 0)
    ax.axhline(y=avg_probing, xmin = 0.51, color='black', linestyle=(5, (10, 3)), alpha = 0.5, zorder = 0)
    
    # Add text to the lines
    ax.text(x = -0.1, y = 1.04, s = f'LLM Next Token', color='black', ha='left', va='bottom', alpha = 0.9, fontsize=18,
            bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=0.5', zorder = 100))
    ax.text(x = 0.05, y = 0.961, s = f'Pr@1, avg. {avg_llm:.0%}', color='black', ha='left', va='bottom', alpha = 0.7, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3', pad = 0.1), zorder = 1)

    ax.text(x = 2.6, y = 1.04, s = f'VSA Probing', color='black', ha='center', va='bottom', fontsize=18,
            alpha = 0.9, bbox=dict(facecolor='white', alpha=1, edgecolor='black',  boxstyle='round,pad=0.5', pad = 0.3), zorder = 100)
    ax.text(x = 2.6, y = 0.961, s = f'Pr@1, avg. {avg_probing:.0%}', color='black', ha='center', va='bottom', fontsize=12, alpha = 0.7,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3', pad = 0.1), zorder = 1)
    
    # Plot vertical line, separating the two groups
    ax.axvline(x=1.5, color='black', ymin= -1, linestyle='-', alpha=0.7, clip_on=False)
    
    # Add maxium and minimum values
    maximums = scores[['next_token_precision@1','next_token_precision@5','precision@1', 'precision@5']].max().values
    minimums = scores[['next_token_precision@1', 'next_token_precision@5', 'precision@1', 'precision@5']].min().values
    for bar in ax.patches:
        value = bar.get_height()
        pos = bar.get_x() + bar.get_width() / 2
        
        if value in maximums:
            marker='^'
        elif value in minimums:
            marker='v'
        else:
            continue
        ax.scatter(x = pos, y = value, marker=marker, s=100, zorder=5, alpha = 0.5,  color='black', edgecolor='black')
        ax.text(x = pos, y = value + 0.02, s = f'{value:.0%}', ha='center', va='bottom', fontsize=10, color='black', alpha = 0.8, 
                bbox=dict(facecolor='white', alpha=0.4, edgecolor = 'none', boxstyle='round,pad=0.2'), zorder = 1) #  edgecolor='black',
   
    # Set the axes labels and title
    plt.ylabel('Precision@K', fontsize = 20)
    plt.ylim(0, 1)
    plt.xlabel('')
    plt.yticks(fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(visible=True, which='minor', axis='y', linestyle='-', linewidth=0.3, color='lightgray', alpha = 1)
    
    # Legend
    legend = plt.legend(title = 'LLMs', ncols = 1, loc='upper right', fontsize = 12, title_fontsize = 16, columnspacing = 1, 
                        bbox_to_anchor = (1.19, 0.75), facecolor = 'whitesmoke', edgecolor = 'darkgray', framealpha = 0.5)
    for handle in legend.legend_handles:
        handle.set_edgecolor('none')
    
    # Set the x-ticks to be the same as the original
    ax.set_xticks(ax.get_xticks())
    labels = [item.get_text().replace('_', ' ').replace('next token ', '').replace('precision', 'pr').capitalize() for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=16)
    
    # Figure aesthetics
    sns.despine(left = True, bottom= True)
    plt.tight_layout(pad = 0, rect=(0, 0.005, 1, 0.96))
    
    # Save the figure
    plt.savefig(path.join(root_folder, 'precision_scores.pdf'), dpi = 400)


def load_data(folder):
    
    name_mapping = {
        'Llama3': 'Llama 3.1',
        'Llama4': 'Llama 4',
        'Gpt2': 'GTP-2',
        'Phi4': 'Phi-4',
        'Olmo2':'OLMo-2'
    }
    
    # For all foder read the precision_score json file
    average_scores = defaultdict(list)
    
    for folder in listdir(root_folder):
        
        # Skip the random and non-existing folders
        if 'random' in folder or not path.isdir(path.join(root_folder, folder)):
            continue
        
        # Load and save the averaged scores
        with open(path.join(root_folder, folder, 'descriptive_stats', 'precision_scores.json'), 'r') as file:
            precision_score = json.load(file)
        for k, v in precision_score.items():
            average_scores[k].append(v)
            
        # Save the LLM name
        average_scores['LLM'].append(folder.split('_')[0].capitalize())
        average_scores['LLM'] = [name_mapping[llm] if llm in name_mapping else llm for llm in average_scores['LLM']]
        
    # Convert to a DataFrame
    average_scores = pd.DataFrame(average_scores)
    return average_scores

if __name__ == '__main__':
    root_folder = path.join('outputs', 'probing', 'stats')

    # Load the data
    scores = load_data(root_folder)

    # Create the graph 
    create_graph(scores)

    # Save the data
    scores.describe().round(3).to_json(path.join(root_folder, 'avg_precision_scores.json'), indent=4)
    
    print('DONE')