from collections import defaultdict
from os import makedirs, path
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Llama4ForConditionalGeneration
import json
import re
import numpy as np
import pandas as pd
import torch
import seaborn as sns

def load_llm(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map = 'auto')
    
    # Set the model to evaluation mode
    model.eval()
    
    # Print the model and tokenizer details
    print('Loaded model:', model_name, 'device:', model.device)
    
    return model, tokenizer

def logit_lens(hs, projector, tokens, target_word, pos = 0, create_plot = True):
    
    # Stack the hidden states
    hs = torch.stack(hs).squeeze()
    
    outputs = list()
    for layer_pos, layer_embeddings in enumerate(hs):
        
        # Example: Process all tokens in a layer at once
        logits = projector(layer_embeddings)  # Project all token embeddings at once
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(3, dim=-1)
        
        for token_pos, (token_probs, token_indices) in enumerate(zip(top_probs, top_indices)):
            top_tokens = [tokenizer.decode([idx]) for idx in token_indices]
            top_tokens = list(zip(top_tokens, token_probs.tolist()))
            outputs.append({'token': token_pos, 'layer': layer_pos, 'top_token': top_tokens[pos][0], 'prob': top_tokens[pos][1]})

    # Plot the results (X: layers Y: tokens, value = prob)
    outputs = pd.DataFrame(outputs)
    
    if create_plot:
        
        # Clean the tokens    
        cleaned_tokens = [t.strip('Ġ').lower() for t in tokens]
        
        # Filter the outputs
        median_layer = hs.shape[0] // 2
        outputs = outputs[outputs['layer'] >= median_layer]
        num_layers = outputs['layer'].nunique()
        
        # Pivot Data to Create Heatmap Structure
        heatmap_data = outputs.pivot(index="layer", columns="token", values="prob").to_numpy()
        token_labels = outputs.pivot(index="layer", columns="token", values="top_token").to_numpy()
    
        # Create Heatmap
        fig, ax = plt.subplots(figsize=(13, num_layers // 3))
        sns.heatmap(heatmap_data, annot=False, cmap="Reds", linewidths=1, linecolor='black', ax=ax, vmin=0, vmax=1, cbar_kws={'shrink': 0.3, 'format': '%.1f'})
        
        # Example: Use a single loop for text overlay
        for i, (row, tokens) in enumerate(zip(heatmap_data, token_labels)):
            for j, (prob, token) in enumerate(zip(row, tokens)):
                
                # Ensure the token is valid
                if not isinstance(token, str):
                    token = '-'
                if token.strip() == '':
                    token = 'SPACE'
                token = clean_token(token[:10]) if isinstance(token, str) else 'SPACE'

                # Set text color based on probability
                text_color = "white" if prob > 0.5 else "black"
                
                if token.strip().lower() in cleaned_tokens + [target_word]:
                    text_color = "yellow" if prob > 0.5 else "firebrick"
                
                # Add text to the heatmap
                try:
                    plt.text(
                        j + 0.5, i + 0.5, token, 
                        ha='center', va='center', 
                        color=text_color, fontsize=10, weight="bold")
                except Exception as e:
                    print(f"Error rendering token '{token}': {e}")

        # Format Axes
        ax.set_xticks(np.arange(len(heatmap_data[0])) + 0.5)
        ax.set_xticklabels(cleaned_tokens, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(np.arange(len(heatmap_data)) + 0.5)
        ax.set_yticklabels(sorted(outputs['layer'].unique()), fontsize=12, rotation = 0)
        ax.set_xlabel("Token", fontsize=20, color ='firebrick')
        ax.set_ylabel("Layer", fontsize=20, color ='firebrick')
        #ax.set_title("Model's Top Token with its SoftMax score", fontsize=28, weight="bold", color ='firebrick', pad = 1)

        # Add Color Bar
        cbar = ax.collections[0].colorbar
        cbar.set_label("Softmax", fontsize=20, rotation=90) # , labelpad=-15
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.set_label_position('left')
        
        try:
            fig.tight_layout(pad = 0)
        except Exception as e:
            pass
    
        return outputs, fig
    else:
        return outputs, None
    
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

def clean_token(token):
    try:
        return ''.join(c if ord(c) < 128 else '?' for c in token)
    except TypeError:
        return '-'

if __name__  == '__main__':
    print(f'GPUs ({torch.cuda.device_count()}):\n' + '\n'.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    
    with open(path.join('data', 'verbose_examples.json'), 'r') as file:
        data = json.load(file)
    #prompts = [item for items in data.values() for item in items]
    
    areas = create_macro_areas(data.keys())
    
    #prompts = np.random.default_rng(seed = 101).choice(prompts, 2).tolist()
    
    # Save the prompt domains
    #prompt_domains = {p: domain for p in prompts for domain, docs in data.items() if p in docs}
    
    # Load the model
    model, tokenizer = load_llm(model_name = 'microsoft/phi-4') # meta-llama/Llama-3.1-8B || allenai/OLMo-2-0325-32B || meta-llama/Llama-4-Scout-17B-16E
    median_layer = model.config.num_hidden_layers // 2
    
    if hasattr(model, 'lm_head'):
        unembedding_layer = model.lm_head
    elif hasattr(model, 'embed_out'):
        unembedding_layer = model.embed_out
    else:
        raise ValueError(f"Unsupported model architecture: {model.config.architectures[0]}")
    
    # Define the prompts
    prompts = [
        " chile is to santiago as albania is to tirana",
        " expire is to expiration as standardize is to standardization",
        " paid is to unpaid as suitable is to unsuitable",
        " full is to empty as introvert is to extravert",
        " france is to paris as libya is to 5ripoli",
        " small is to big as extravert is to introvert",
        " daughter is to son as superwoman is to superman",
        " selling is to sold as describing is to described",
        " category is to categories as story is to stories",
        " chile is to spanish as iran is to persian",
        " hume is to philosopher as strauss is to composer",
        " computation is to compute as illumination is to illumine",
        " baghdad is to iraq as georgetown is to guyana"
    ]
    
    #data = {'capital_world': prompts}
    
    # Create the output folder
    output_folder = path.join('outputs', 'lens_gpt2')
    makedirs(output_folder, exist_ok = True)
    
    generated_tokens, metrics = [], []
    for dom_counter, (domain, docs) in enumerate(data.items()):
        #docs = np.random.default_rng(seed = 101).choice(docs, 2).tolist()
        for i, text in enumerate(tqdm(docs,  desc = f'Processing {domain} ({dom_counter + 1}/{len(data.keys())})')):

            parts = text.split()
            partial_doc = ' '+ ' '.join(parts[:-1])
            target_word = parts[-1].lower()
            words = [word.lower().strip() for word in parts if word not in ['is', 'to', 'as']]
            
            # Tokenize the text
            inputs = tokenizer(partial_doc, return_tensors="pt").to(model.device)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states = True)
            
            # Logit Lens
            outputs, fig = logit_lens(outputs.hidden_states, projector=unembedding_layer, 
                                      tokens = tokens, target_word = target_word,
                                      pos = 0, create_plot = True if text.lower() in prompts else False)
            
            if fig is not None:
                try:
                    fig.savefig(path.join(output_folder, f'{domain}_doc{i + 1}.pdf'))
                    plt.close(fig)
                except Exception as e:
                    print(f"Error saving figure for {domain} doc {i + 1}: {e}")
                
            # Extract the next token for each layer of the considered token
            last_token = outputs['token'].max()
            
            # Example: Use NumPy for filtering
            cond = (outputs['token'] == last_token) & (outputs['layer'] >= median_layer)
            outputs = outputs[cond].reset_index(drop=True)
            outputs['token'] = outputs['token'].map(lambda token_pos: tokens[token_pos])
            generated_tokens.append({text:outputs.to_dict(orient = 'records')})

            # Compute the metrics
            part_mapping = {0: 'Example_key', 1: 'Example_value', 2: 'Key', 3: 'Target'}
            labeled_words = {part.lower(): part_mapping[i] for i, part in enumerate(words)}
            generated_tokens = outputs['top_token'].unique().tolist()
            generated_tokens = [token.lower().strip() for token in generated_tokens]  
            
            # Fuzzy match function
            fuzzy_match = lambda token, word: token == word or word.startswith(token)
            
            # Extract concepts
            extracted_concepts = set()
            for token in generated_tokens:
                if len(token) > 1:
                    matches = [label for word, label in labeled_words.items() if fuzzy_match(token, word)]
                    extracted_concepts.update(matches)
                    
            # Remove duplicates and sort
            extracted_concepts = sorted(extracted_concepts)
            
            if 'Example_key' in extracted_concepts and 'Example_value' in extracted_concepts:
                extracted_concepts.remove('Example_key')
                extracted_concepts.remove('Example_value')
                extracted_concepts.append('Example')

            # Compute precision@1
            precision_1 = 'Target' in extracted_concepts
        
            # Save the metrics
            metrics.append({
                'domain':domain,
                'area': areas[domain],
                'prompt': partial_doc,
                'target': target_word,
                'generated_tokens': list(set(generated_tokens)),
                'extracted_concepts': extracted_concepts,
                'precision@1': precision_1,
            })
            
# Compute the average metrics
metrics_df = pd.DataFrame(metrics)
metrics_df['extracted_concepts'] = metrics_df['extracted_concepts'].map(lambda concepts: '|'.join(concepts) if len(concepts) > 0 else 'NONE')

# Extracted concepts
extracted_concepts = metrics_df['extracted_concepts'].value_counts(normalize=True)

# Extracted concepts by group
concepts_byGroups = defaultdict(dict)
for col in ['precision@1', 'area', 'domain']:
    for area in metrics_df[col].unique():
        area_concepts = metrics_df[metrics_df[col] == area]
        counter = area_concepts['extracted_concepts'].value_counts(normalize = True).sort_values(ascending=False).round(3)
        concepts_byGroups[col][str(area)] = counter.to_dict()

avg_metrics = {
    'avg_precision@1':  metrics_df['precision@1'].agg(['mean', 'std', 'count']).round(3).to_dict(),
    'extracted_concepts': extracted_concepts.round(3).to_dict(),
    'extracted_concepts_byGroups': concepts_byGroups,
}

# Print the average metrics
print('Average metrics:\n', json.dumps(avg_metrics, indent=4))

# Save the metrics to JSON files
with open(path.join(output_folder, 'avg_metrics.json'), 'w') as file:
    json.dump(avg_metrics, file, indent=4)
with open(path.join(output_folder, 'extracted_concepts.json'), 'w') as file:
    json.dump(metrics, file, indent=4)