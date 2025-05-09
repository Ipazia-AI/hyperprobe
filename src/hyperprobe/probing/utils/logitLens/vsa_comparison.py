import json
from os import path
import pandas as pd

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
    logit_stats = pd.read_json(path.join('outputs', f'lens_{model_name}', 'extracted_concepts.json'))
    logit_stats['prompt'] = logit_stats['prompt'].str.strip()
    
    # Import the VSA stats  
    vsa_stats = load_vsa_stats(path_file = path.join('outputs', 'probing', f'{model_name}_13apr_verbose.json'))
    
    # ANALYSIS 1: Empty representation for LOGIT
    empty_logit = logit_stats[logit_stats['extracted_concepts'].apply(len) == 0]
    vsa_stats_corresponding = vsa_stats[vsa_stats['doc'].str.lower().isin(empty_logit['prompt'].str.lower())]
    vsa_extracted_concepts = vsa_stats_corresponding['extracted_factors'].value_counts(normalize=True).round(3)
    print(f"VSA-based extaction corresponding to empty logit ({len(empty_logit)}):\n", vsa_extracted_concepts)    