import json
from os import path
import pandas as pd

from hyperprobe.statistics.utils import create_macro_areas

def load_vsa_stats(path_file: str) -> pd.DataFrame:    
    """
    Load the VSA stats from a JSON file.
    """
    with open(path_file, 'r') as f:
        vsa_stats = json.load(f) 
    all_docs = []
    for domain, docs in vsa_stats.items():
        for doc in docs:
            
            if not doc:
                continue

            # Get the logit stats for the current doc
            doc['domain'] = domain
            
            # Expand the precisions key into separate columns
            doc = doc | doc['precisions']
            
            # Remove the precisions key from the dictionary
            doc.pop('precisions')
            
            # Add the logit stats to the doc    
            all_docs.append(doc)
    return pd.DataFrame(all_docs)

if __name__ == "__main__":
    
    model_name = "llama4"

    # Import the logit stats
    logit_folder = path.join('original_outputs', 'lens', model_name)
    logit_stats = pd.read_json(path.join(logit_folder, 'extracted_concepts.json'))
    logit_stats['prompt'] = logit_stats['prompt'].str.strip()
    
    # Import the VSA stats  
    vsa_stats = load_vsa_stats(path_file = path.join('original_outputs', 'probing', f'{model_name}_13apr_verbose.json'))
    
    # ANALYSIS 1: Empty representation for LOGIT
    empty_logit = logit_stats[logit_stats['extracted_concepts'].apply(len) == 0]
    vsa_stats_corresponding = vsa_stats[vsa_stats['doc'].str.lower().isin(empty_logit['prompt'].str.lower())]
    vsa_extracted_concepts = vsa_stats_corresponding['extracted_factors'].value_counts(normalize=True).round(3)
    print(f"VSA-based extaction corresponding to empty logit ({len(empty_logit)}):")
    
    # Group by domain and print the counts
    area_mapping = create_macro_areas(vsa_stats_corresponding['domain'].unique())
    vsa_stats_corresponding['area'] = vsa_stats_corresponding['domain'].map(area_mapping)
    vsa_extracted_concepts_byDomain = vsa_stats_corresponding.groupby('area')['extracted_factors'].value_counts(normalize=True).round(3)
    
    # Analysis 2: Empty representation for VSA
    empty_vsa_stats = vsa_stats[vsa_stats['precision@1'] == 0]
    logit_corresponding = logit_stats[logit_stats['prompt'].str.lower().isin(empty_vsa_stats['doc'].str.lower())]

    logit_extracted_concepts = logit_corresponding['extracted_concepts'].apply('|'.join).value_counts(normalize=True).round(3)
    
    # Save the results
    logit_extracted_concepts = {model_name: logit_extracted_concepts.to_dict()}
    
    # Save the results to a JSON file
    json_file = path.join('empty_vsa_logit_stats.json')
    if path.exists(json_file):
        with open(json_file, 'r') as f:
            stats = json.load(f)
        logit_extracted_concepts = stats | logit_extracted_concepts
    with open(json_file, 'w') as f:
        json.dump(logit_extracted_concepts, f, indent=4)   
    
    df = pd.DataFrame(logit_extracted_concepts)
    
    print(df)
    df.to_latex(path.join('empty_vsa_logit_stats.tex'), index = False, escape = False)
    

    # Save the subset of instances
    with pd.ExcelWriter(path.join(logit_folder, 'empty_logit_vsa_stats.xlsx')) as writer:
        vsa_stats_corresponding.to_excel(writer, sheet_name='Overall', index = False)
        vsa_extracted_concepts_byDomain.to_excel(writer, sheet_name='Group by domain')
        
        
        #for domain, group in vsa_stats_corresponding.groupby('domain'):
        #    group.to_excel(writer, sheet_name=domain, index=False)