from os import makedirs, path
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import json
import numpy as np
import pandas as pd
import seaborn as sns
import re
import textwrap

# CONSTANTS
precision_mapping = {0: 'FALSE', 0.5: 'Initial\ntoken', 1: 'TRUE'}

# Set the style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

def create_macro_areas(domains):
    
    patterns = {
        "Factual Knowledge": r"(capital|currency|city|country|occupation|country_|name_occupation|name_nationality|city_county|color)",
        "Semantic Hierarchies": r"(hyponyms|hypernyms|meronyms|member|substance)",
        "Semantic Relations": r"(synonyms|antonyms|opposite|animal|family|male_female|nationality)",
        "Morphological Modifiers": r"(adj\+ly_reg|noun\+less_reg|verb\+ment_irreg|adj\+ness_reg|verb\+tion_irreg|verb\+able_reg|verb\+er_irreg|over\+adj_reg|un\+adj_reg|re\+verb_reg|\+|over\+|un\+|re\+|able|less|tion|ment|ness|lative|adj_to_adverb|superlative|comparative)",
        "Verbal & Grammatical Forms": r"(verb(?:.*3pSg|.*Ving|.*Ved)?|adj.*|noun.*|tense|3pSg|Ving|Ved|plural|present)",
        "Mathematics": r"(math)"
    }
    
    areas = dict()
    for domain in domains:
        for category, pattern in patterns.items():
            if re.search(pattern, domain):
                if domain in areas and areas[domain] != category:
                    print(f'{domain} -> {category} (overwriting {areas[domain]})')
                areas[domain] = category
                break
    
    return areas

    
def create_graphs(df, output_folder):
    
    graph_folder = path.join(output_folder, 'graphs')
    makedirs(graph_folder, exist_ok = True)
    
    # Cols to exclude
    df = df.drop(columns = ['target_token_softmax', 'normalized_target_token_rank', 'precision@3'])
    
    # Select the numerical columns
    numerical_cols = df.select_dtypes(exclude = 'object').columns
    numerical_cols = sorted(numerical_cols, key = lambda col: ('token' not in col, 'precision' not in col))

    # Create the feature-wise figure
    ncols = 3
    nrows = int(np.ceil(len(numerical_cols) / ncols))
    
    # COLORS 
    colors = sns.color_palette("tab10", n_colors = len(df['area'].unique()))
    area_colors = {area: colors[idk] for idk, area in enumerate(df['area'].unique())}
    df['color'] = df['area'].map(lambda area: area_colors[area])
    domain_colors = df[['domain', 'color']].drop_duplicates().set_index('domain')['color'].to_dict()
    
    # Create the graphs
    for type in df['unbunding_type'].unique().tolist() + ['ALL'] + ['WRONG']:
        
        if type == 'ALL':
            df_type = df
        elif type == 'WRONG':
            df_type = df[df['next_token_precision@5'] == 0]
        else:
            df_type = df[df['unbunding_type'] == type]
            
        # Ket the domain-aree pairs
        unique_domains = df_type[['domain', 'area']].drop_duplicates()
        
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(15, 20)) # 
        axes = axes.flatten()
        for i, col in enumerate(numerical_cols):
            
            if 'precision' in col:

                # Map the precision values
                df_type = df_type.sort_values(by = col)
                df_type[col] = df_type[col].map(lambda x: precision_mapping[x])
                
                # Sort the hue values
                sorted_areas_list = df_type.groupby([col, 'area']).size().reset_index(name='count')\
                    .groupby('area')['count'].mean().sort_values(ascending=False).index.tolist()
                area_rank_map = {area: rank for rank, area in enumerate(sorted_areas_list)}

                unique_domains['rank'] = unique_domains['area'].map(area_rank_map)
                sorted_domain = unique_domains.sort_values(by='rank', ascending=False)['domain'].tolist()
       
                # Create the histogram for the categorical data
                sns.histplot(data = df_type, x = col, hue = 'domain', hue_order = sorted_domain, ax = axes[i], 
                             bins = len(df[col].unique()), discrete=True, multiple='stack', stat = 'proportion', element="bars",
                             binwidth=0.5, palette = domain_colors, edgecolor="none", legend=False)
                         
                # Add text values above each bar
                percentages = df_type[col].value_counts(normalize = True)
                for bin_name in df_type[col].unique():
                    axes[i].text(x = bin_name, y = np.clip(percentages[bin_name] + 0.04, a_min = 0, a_max = 0.97), 
                                 s = f'{percentages[bin_name]:.1%}', 
                                 ha = 'center', va = 'bottom', fontsize = 16, color = 'black',
                                 bbox=dict(facecolor='white', alpha=0.5, edgecolor = (0.5, 0.5, 0.5, 0.5), boxstyle='round,pad=0.3'))
                    
                # Add average line
                avg = df_type[col].value_counts(normalize = True).mean()
                
                # Set the y ticks
                axes[i].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, axis='y', alpha = 0.3)
                axes[i].grid(False, axis='x')
                
                axes[i].set_ylabel('Observations', fontsize = 16)
            else:
                
                # Sort the hue values
                sorted_areas_list = df_type[['area', col]].groupby('area').mean().sort_values(by = col, ascending=False).index.tolist()
                area_rank_map = {area: rank for rank, area in enumerate(sorted_areas_list)}

                unique_domains['rank'] = unique_domains['area'].map(area_rank_map)
                sorted_domain = unique_domains.sort_values(by='rank', ascending=True)['domain'].tolist()
                
                # Create the boxplot for the numerical data
                #sorted_domain = unique_domains.sort_values(by = ['area'])['domain'].tolist()
                sns.boxplot(data = df_type, x = col, hue = 'domain', hue_order= sorted_domain, ax = axes[i], palette = domain_colors, legend=False, dodge=0.5)
                #violinplot || boxplot
                
                # Add the average line
                avg = df_type[col].median().round(2) 
                label = f"Median ({int(avg) if 'rank' in col else avg})"
                axes[i].axvline(avg, color='black', linestyle='--', linewidth = 1, 
                                label = label)
                axes[i].legend(loc='lower right', fontsize=16)
                
                axes[i].set_ylabel('Domains', fontsize = 16)
                
                # Set the x ticks
                axes[i].grid(False, axis='y')
                axes[i].grid(True, axis='x', alpha = 0.3)
                q_high = df_type[col].quantile(q = 0.99)
                axes[i].set_xlim(left = 0, right = max([1, q_high]))
            
            # Set the title and labels
            title = col.replace('_', ' ').upper()
            if len(title) >= 15:
                title = textwrap.fill(title, width=15)
            axes[i].set_title(title, color = 'black', fontsize = 30, pad = 20)
            axes[i].set_xlabel('')
            
            # Set the ticks
            axes[i].tick_params(axis='both', which='major', labelsize=20)
        
        # Remove empty graphs
        empty_graphs = len(axes) - len(numerical_cols)
        if empty_graphs > 0:
            for i in range(1, empty_graphs + 1):
                axes[-i].axis('off')
        
        # Add figure-level legend
        area_colors = dict(sorted(area_colors.items(), key = lambda x: len(df.loc[df['area'] == x[0], 'domain'].unique()), reverse = True))
        fig.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label = area + f" ({len(df.loc[df['area'] == area, 'domain'].unique())})")
                for area, color in area_colors.items()],
            loc='lower right',
            bbox_to_anchor=(1, 0.08),
            ncols = 1,
            fontsize=18,
            title=f"AREAS ({len(df['domain'].unique())} domains)",  
            labelcolor = '#2e2e2e', 
            title_fontsize=18
        )
        
        #fig.supylabel('Feature', fontsize=40)
        fig.tight_layout(h_pad=2) # rect=[0, 0.08, 0.99, 1],
        fig.savefig(path.join(graph_folder, f'{type}.pdf'))  
    
    # Create the domain folder
    domain_folder = path.join(graph_folder, 'domains')
    makedirs(domain_folder, exist_ok = True)
    for domain in df['domain'].unique():
        
        # Select the data
        domain_df = df[df['domain'] == domain]
        
        # Create the histograms
        ncols = 3
        nrows = int(np.ceil(len(numerical_cols) / ncols))
        
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20, 10))
        axes = axes.flatten()
        for i, col in enumerate(numerical_cols):
            sns.histplot(data = domain_df, x = col, stat = 'percent', kde = False, bins = 50, alpha = 0.6, ax = axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Percent')
            axes[i].grid(True, alpha = 0.3)
        fig.tight_layout()
        fig.savefig(path.join(domain_folder, f'{domain}_hist.pdf'))
        plt.close(fig)

def plot_corr(corr):
    
    # Drop the rows with all NaN values
    corr = corr.dropna(axis = 1, how = 'all')
    corr = corr.dropna(axis = 0, how = 'all')
    
    if corr.shape[0] == 0:
        return None
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))
        
    # Create the heatmap
    sns.heatmap(corr, cmap='coolwarm', center=0, vmax=1, vmin= -1, 
                annot=True, fmt='.1f', square=True, 
                #mask = np.tril(np.ones_like(corr.corr())), 
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.7, 'pad': 0.15,
                          'format': '%.1f', 'ticks': [-1, -0.5, 0, 0.5, 1]},
                ax = ax)
    
    # Rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Move the colorbar to the left
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_ticks_position('left')
    
    # Set the title
    ax.set_title('Spearman Correlation')
    
    # Tight layout
    fig.tight_layout()
    
    # Save and close
    plt.close(fig)
    
    return fig 
       
def compute_correlations(df, output_folder):
    
    # Create the output folder
    output_folder = path.join(output_folder, 'correlations')
    makedirs(output_folder, exist_ok = True)
    
    # Sort and rename the columns
    cols = sorted(df.columns, key = lambda col: ('token' not in col, 'precision' in col))
    df = df[cols]
    df.columns = df.columns.map(lambda x: x.replace('_', ' '))

    # Overview (spearman, focus on monotonic Relationship)
    fig = plot_corr(df.select_dtypes(exclude = 'object').corr(method = 'spearman'))
    fig.savefig(path.join(output_folder, 'corr.pdf'))
    
    # Compute the correlations for each column
    for col in ['precision@1', 'next token precision@1', 'next token precision@5', 'area', 'unbunding type']:
        for label in df[col].unique():
            
            # Select the data
            df_label = df[df[col] == label].drop(columns = col)
            df_label = df_label.select_dtypes(exclude = 'object')
            
            # Create the correlation matrix
            fig = plot_corr(df_label.corr(method = 'spearman'))
            
            if fig is None:
                continue
            
            # Create the output folder
            output_folder_label = path.join(output_folder, col)
            makedirs(output_folder_label, exist_ok = True)
            
            # Save the figure
            labelName = precision_mapping[label].lower() if isinstance(label, float) else ''.join([word.capitalize() for word in str(label).split()])
            fig.savefig(path.join(output_folder_label, f'corr_{labelName}.pdf'))

def load_results(results):
    
    # Create dataframe
    dfs = []
    for domain, outputs in results.items():
        
        # Check if the doc is empty
        outputs = [out for out in outputs if out]
        
        # Create the dataframe
        domain_df = pd.DataFrame(outputs)
    
        # Add the domain 
        domain_df['domain'] = domain
        
        # Expand columns 
        cols = [
            domain_df['precisions'].apply(pd.Series),
            domain_df['next_token'].apply(pd.Series).rename(columns = {'token': 'generated_token', 'softmax': 'next_token_softmax', 'correct': 'next_token_precision@1'}),
            domain_df['target_token_rank'].map(lambda x: list(x.values())[0]).apply(pd.Series).rename(columns = {'rank': 'target_token_rank', 'normalized_rank' : 'normalized_target_token_rank','softmax': 'target_token_softmax', 'softmax_diff': 'target_token_softmax_diff'}),
        ]
        dfs.append(pd.concat([domain_df.drop(columns = 'target_token_rank'), *cols], axis = 1).drop(columns = ['precisions', 'next_token']))
        
    # Concatenate the dataframes
    df = pd.concat(dfs)
    
    # Convert to float
    df['next_token_precision@1'] = df['next_token_precision@1'].astype(float)
    
    # Check if target was in the other
    df['next_token_precision@5'] = df.apply(lambda row: 
        max(
            [1 if row['target'].lower() == token.lower() 
            else 0.5 if token and row['target'].lower().startswith(token.lower()) 
            else 0 
            for token in row['top5_tokens'].keys()] or [0]
        ), axis=1)
    
    # Add the macro-areas
    areas = create_macro_areas(df['domain'].unique())   
    df['area'] = df['domain'].map(lambda domain: areas[domain])
    df = df.sort_values(by = 'area').reset_index(drop = True)
    
    # Sort the columns
    relevant_order = ['area', 'domain', 'doc', 'target', 
                      'generated_token', 'next_token_softmax', 'next_token_precision@1', 'next_token_precision@5','top5_tokens',
                      'target_token_rank', 'target_token_softmax', 'target_token_softmax_diff', 'normalized_target_token_rank',
                      'unbunding_type', 'extracted_factors', 'target_vsa_cosine_sim', 
                      'precision@1', 'precision@3', 'precision@5', 'best_item_type','vsa_sim']
    cols = sorted(df.columns, key = lambda x: relevant_order.index(x) if x in relevant_order else 1000)
    df = df[cols].drop(columns = ['unbunding_combo'])

    print('DATA:', len(df), '--> dropped columns:', ' | '.join(set(df.columns) - set(relevant_order)))

    return df

def descriptive_stats(df, output_folder):
    
    # Create the output folder
    output_folder = path.join(output_folder, 'descriptive_stats')
    makedirs(output_folder, exist_ok = True)
    
    col_sort_cond = lambda col: ('token' in col, 'precision' not in col)
    cols = sorted(df.columns, key = col_sort_cond)
    df = df[cols]
    
    # Compute the stats based on the domain
    domain_stats = df.groupby(by = ['unbunding_type', 'domain']).describe().round(2)
    with pd.ExcelWriter(path.join(output_folder, 'domain_stats.xlsx')) as writer:
        for group in domain_stats.index.get_level_values(0).unique():
            to_save = domain_stats.loc[group].sort_values(by = ('target_vsa_cosine_sim', 'mean'), ascending = False)
            to_save.to_excel(writer, sheet_name = group)
    
    # Compute the stats based on the domain
    area_stats = df.groupby(by = ['unbunding_type', 'area']).describe().round(2)
    with pd.ExcelWriter(path.join(output_folder, 'area_stats.xlsx')) as writer:
        for group in area_stats.index.get_level_values(0).unique():
            to_save = area_stats.loc[group].sort_values(by = ('target_vsa_cosine_sim', 'mean'), ascending = False)
            to_save.to_excel(writer, sheet_name = group)
    
    # Compute the stats based on the model precision
    precision_stats = df.groupby(by = ['unbunding_type', 'next_token_precision@1']).describe().round(2)
    with pd.ExcelWriter(path.join(output_folder, 'precision_stats.xlsx')) as writer:
        for group in precision_stats.index.get_level_values(0).unique():
            to_save = precision_stats.loc[group].sort_values(by = ('target_vsa_cosine_sim', 'mean'), ascending = False)
            to_save.to_excel(writer, sheet_name = group)
            
    # Compute the stats based on the extracted factors
    prec_cols = [col for col in df.columns if 'precision' in col]
    probe_cols = ['area', 'domain']
    for probe_col in probe_cols:
        with pd.ExcelWriter(path.join(output_folder, f'extracted_factors_by{probe_col.capitalize()}.xlsx')) as writer:
            for value in df[probe_col].unique():
            
                # Select the data
                stats = pd.concat([df.loc[df[probe_col] == value, col].astype('object').value_counts(normalize = True).rename(col) for col in prec_cols], axis = 1)
                stats = stats.fillna(0).T
                
                # Save the data
                stats.to_excel(writer, sheet_name = value)
    
    # Save the precision scores
    avg_metrics = df.select_dtypes(exclude = 'object').agg(['mean', 'std']).round(3)
    print(avg_metrics)
    with open(path.join(output_folder, 'precision_scores.json'), 'w') as file:
        json.dump(avg_metrics.to_dict(), file)
    

def view_observations(to_view, name, output_folder):
    
    # Sort the data
    to_view = to_view.sort_values(by = ['target_vsa_cosine_sim', 'unbunding_type', 'next_token_precision@5','next_token_softmax'], ascending = False)
    
    # Compute the stats
    precision_stats = pd.DataFrame({col: to_view[col].value_counts(normalize=True) for col in to_view.select_dtypes(include=['bool']).columns}).T

    with pd.ExcelWriter(path.join(output_folder, f'{name}.xlsx')) as writer:
        
        # Save the data
        to_view.to_excel(writer, index = False, sheet_name = 'items')
        
        # Save the numerical stats
        to_view.describe().T.to_excel(writer, sheet_name = 'num_stats')

        # Save the categorical stats
        precision_stats.to_excel(writer, sheet_name = 'precision_stats')

        # Save the categorical stats
        for col in ['area', 'domain', 'unbunding_type', 'extracted_factors', 'generated_token', 'next_token_precision@5', 'best_item_type']:
            to_view[col].value_counts(normalize=True).to_excel(writer, sheet_name = f'{col}_stats')