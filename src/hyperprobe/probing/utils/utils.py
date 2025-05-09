from os import path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import itertools
import json
import re
import numpy as np
import pandas as pd
import torch

# LOCAL IMPORTS
from hyperprobe.encoder.utils.encoder import VSAEncoder
from hyperprobe.data_creation.utils.emb_utils import kmeans_cuda

def load_llm(model_name: str, dtype: torch.dtype = torch.bfloat16, device = None):

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = dtype, device_map = 'auto')

    # Set the token id for padding
    model.pad_token_id = tokenizer.eos_token_id

    # Set the model to evaluation mode
    model.eval()
    
    # Clear the cache
    torch.cuda.empty_cache()
    
    print(f"\nLoaded the LLM ({model_name}, {dtype}, {model.device} --> {torch.cuda.get_device_name(model.device)})\n")
    return tokenizer, model

def load_vsaTranslator(model_name, device):
    
    # Set the base path
    root_path = path.join('outputs', 'adapter', 'models')

    # Load the model
    model = VSAEncoder.load_from_checkpoint(path.join(root_path, model_name), map_location = device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"VSA translator ({model_name}) has {num_params / 1e6:.2f}M parameters")
    
    # Set the model to evaluation mode
    model.eval()

    # Clear the cache
    torch.cuda.empty_cache()
    
    print(f"Loaded the VSA translator ({num_params / 1e6:.0f}M) from '{model_name}' (device: {device})")
    
    return model

def generate_vsa(doc, llm, translator):
    
    # Get the model and tokenizer
    tokenizer, model = llm

    # Tokenize the input
    inputs = tokenizer(doc, return_tensors="pt").to(model.device)
    
    # Generate the LLM embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states = True, output_logits = True, temperature = 0, max_new_tokens = 1)

    # Probe the next token
    token_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()[-1]
    
    # Select the relevant tokens
    #tokens = np.array([t.strip('Ġ') for t in tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze())])
    token_positions = [-1]
    
    # Get the embeddings of the last token for the second half of the hidden layers
    embeddings = torch.stack(outputs.hidden_states).squeeze()
    hs = embeddings[16:, token_positions].squeeze()
    
    # Extract the relevant embeddings
    _, centroids = kmeans_cuda(hs, K = 5)
    
    # Flatten the centroids
    hs = centroids.sum(dim = 0).unsqueeze(0)
    
    # Randomly permute 
    # #hs = hs[:, torch.randperm(hs.shape[1], device=hs.device)]

    assert hs.size(1) == translator.input_dim, f"ERROR: The size of the embeddings ({hs.size(0)}) is not equal to the input dimension of the VSA translator ({translator.input_dim})"
    
    # Convert the embeddings to float and move to the translator's device
    hs = hs.float().to(translator.device)
    
    # Generate the VSA encoding from the embeddings
    with torch.no_grad():
        vsa = torch.sign(translator(hs))
        
    return vsa, token_probs

def unbind(vsa, items):
    out = vsa.clone()
    for item in items:
        out *= item
    return out

def create_queries(items, item_encodings):
    queries = []
    legend = {0: 'example_key', 1: 'example_value', 2: 'key'}
    for r in range(len(items)):
        for combo in itertools.combinations(range(len(items[:-1])), r):
            
            if len(combo) == 0:
                queries.append({'items': [], 'names':[], 'operations': 'vsa', 'type': 'original'})
            else:
                queries.append({
                    'items': [item_encodings[i] for i in combo],
                    'names': [items[i] for i in combo],
                    'operations': 'vsa ⊙ (' + ' ⊙ '.join([items[i] for i in combo]) + ')',
                    'type': 'context' if len(combo) == 3 else 'example' if combo == (0, 1) else '&'.join([legend[i] for i in combo])
                })
                
    # Reverse the query order (from the most specific to the most general)
    queries = queries[::-1]      
            
    return queries

def find_targetToken_in_vocabulary(target, vocabulary):
    target = target.lower()
    positions = []
    for token, pos in vocabulary.items():
        
        # Remove the special token prefix
        token = token.strip('Ġ').lower()
        
        # Check if the token is the target
        if token == target:
            positions.append((token, pos))
        
        # Fuzzy match
        elif len(token) > 1 and target.startswith(token):
            positions.append((token, pos))
    
    positions = sorted(positions, key = lambda x: len(x[0]) / len(target), reverse = True)
    perfect_matches = [item for item in positions if item[0] == target]

    if len(perfect_matches) > 0:
        return perfect_matches
    else:
        return positions
    
def investigate_output_tokens(tokens_softmax, target_ids, tokenizer):
    
    # Sort token indices by descending softmax probability
    ordered_token_ids = torch.argsort(tokens_softmax, descending=True)
    n_tokens = len(ordered_token_ids)
    
    # Define a helper to normalize rank (1 for highest probability, 0 for lowest)
    def normalize_rank(rank):
        return (n_tokens - rank) / n_tokens

    # Find the rank of each target token in target_positions
    raw_target_ranks = []
    for target_token, pos in target_ids:
        indices = torch.where(ordered_token_ids == pos)[0]
        if indices.numel() > 0:
            rank = indices.item()
            raw_target_ranks.append((target_token, rank))
    
    if not raw_target_ranks:
        return None, None
    
    # Select the target token with the smallest (best) rank
    best_target, best_rank = min(raw_target_ranks, key=lambda x: x[1])

    target_info = {
        'rank': best_rank,
        'normalized_rank': normalize_rank(best_rank),
        'softmax_diff': tokens_softmax.max().item() - tokens_softmax[ordered_token_ids[best_rank]].item(),
        'softmax': tokens_softmax[ordered_token_ids[best_rank]].item()
    }
    target_rank = {best_target: target_info}
    
    # Get the top-5 tokens with their softmax scores
    top_probs, top_indices = torch.topk(tokens_softmax, k=5)
    top_tokens = [{
        'token': tokenizer.decode(int(idx)).strip(),
        'softmax': round(prob.item(), 2)
    } for idx, prob in zip(top_indices, top_probs)]
    
    # Return the top token along with a dictionary containing all top-5 tokens
    next_token = top_tokens[0].copy()
    next_token['top5_tokens'] = {token_info['token']: token_info['softmax'] for token_info in top_tokens}
    
    return next_token, target_rank

def probe_doc(doc, codebook, llm, vsa_encoder, pairs = None, solver = None, verbose = True):
    
    # Tokenize the document  
    words = re.split(r"\s[:=]\s", doc.strip())
    words = [word.lower() for word in words]
    
    # Divide the doc into the partial doc and the target answer
    if len(words) > 1:
        cutoff = list(re.finditer(r'\s[:]', doc))[-1].end()
        partial_doc = doc[:cutoff]
        target = doc[cutoff:].strip().lower()
    else:
        tokens = doc.strip().split()
        partial_doc = ' '.join(tokens[:-1])
        target = tokens[-1].lower()
        
        words = [token.lower() for token in tokens if token.lower() in codebook.index]
    assert target in words, f"TARGET NOT IN WORDS: '{target}'"
    
    # Random pairs for unrelated baseline
    #random_keys = np.random.default_rng().choice(list(pairs.keys()), size = 2, replace = False).tolist()
    #random_pairs = [random_keys[0], pairs[random_keys[0]][0], random_keys[1], pairs[random_keys[1]][0]]
    #words = [w.lower() for w in random_pairs]
    #target = words[-1]
    
    # Find the vocabulary position of the target
    target_positions = find_targetToken_in_vocabulary(target, vocabulary = llm[0].get_vocab())
    
    # Check if there is duplicated words (artefacts from the generation of the texts)
    if len(words) != len(set(words)):
        pass

    # Get the VSA encoding for the partial doc
    vsa_doc, tokens_softmax = generate_vsa(partial_doc, llm, vsa_encoder)
    
    # Investigate the output tokens
    next_token, target_rank = investigate_output_tokens(tokens_softmax, target_positions, tokenizer = llm[0])
    
    if next_token is None:
        print("No valid next token found.")
        return {}
    
    # Add the next token the comparison with the target
    next_token['correct'] = next_token['token'].lower() == target    
    if not next_token['correct'] and next_token['token'] and target.startswith(next_token['token'].lower()):
        next_token['correct'] = 0.5
    
    # Find the vsa encoding for all the tokens of the partial doc
    item_encodings = {word: torch.from_numpy(codebook.loc[word].values).to(vsa_doc.device) for word in words if word in codebook.index}
    items = list(item_encodings.keys())
    
    # Unbind the document
    item_encodings = list(item_encodings.values())
    
    # Create the different queries
    queries = create_queries(items, item_encodings)
    
    # Compare the unbinding queries with the target
    best_query, best_sim = None, None
    for query in queries:
        
        # Unbind the document
        unbound_doc = unbind(vsa_doc, query['items'])

        # Compare the unbinding with the target
        sim = pd.Series(
            data = torch.cosine_similarity(
                x1 = unbound_doc, 
                x2 = torch.from_numpy(codebook.values).to(vsa_doc.device)).cpu(),
            index = codebook.index)

        if verbose:
            print(f'QUERY [{query["type"]}, {query["operations"]}] -->', 
                  '| '.join([f'{item.upper() if item == target else item} ({round(sim, 2)})'for item, sim in sim.sort_values(ascending = False).head(5).items()]))
            
        # Check if there is artefact noise after unbinding
        artefact_presence = len(np.intersect1d(sim[sim > 0.1].index, query['names'])) > 0 and sim[target].item() < sim.max()

        # When best_sim is already set, require no artefact interference
        if artefact_presence:
            continue
        
        # skip initialization if best_sim is None and there is artefact noise
        if best_sim is None:
            best_sim = sim.sort_values(ascending=False)
            best_query = query         

        # Evaluate candidate update conditions using descriptive variables
        candidate_is_target_max_and_better = sim[target].item() == sim.max() and sim[target].item() > best_sim[target].item()
        candidate_has_higher_overall_max = sim.max() > best_sim.max() and best_sim[target].item() < best_sim.max()
        candidate_has_equal_max_but_low_second = sim.max() == best_sim.max() and best_sim.iloc[1] < 0.1

        if candidate_is_target_max_and_better or candidate_has_higher_overall_max or candidate_has_equal_max_but_low_second:
            best_sim = sim.sort_values(ascending=False)
            best_query = query
            
    # Greedy stategy (the best similarity is too low)
    if solver is not None and best_sim[target].item() < best_sim.max() and best_sim.max() < 0.5:
        
        # Find the greedy solution
        doc_factors, _ = solver.query(vsa_doc, target = item_encodings[-1])
        
        # Save in case of solution
        if len(doc_factors['features']) > 0 and doc_factors['sim'] > best_sim.max():
            
            # Unbind the document with the greedy solution
            unbound_doc = unbind(vsa = unbound_doc, items = [torch.from_numpy(codebook.loc[feature].values).to(vsa_doc.device) for feature in doc_factors['features']])
            
            # Compute the best similarity
            best_sim = pd.Series(
                data = torch.cosine_similarity(
                    x1 = unbound_doc, 
                    x2 = torch.from_numpy(codebook.values).to(vsa_doc.device)).cpu(),
                index = codebook.index).sort_values(ascending=False)
            
            # Save the best query
            best_query = {
                'names': doc_factors['features'],
                'items': [torch.from_numpy(codebook.loc[feature].values).to(vsa_doc.device) for feature in doc_factors['features']],
                'operations': 'vsa ⊙ (' + ' ⊙ '.join(doc_factors['features']) + ')', 
                'type': 'greedy'}
    
    # Clean up process
    if solver is not None and best_sim.values[0] > 0.1 and best_sim.values[1] > 0.1:
        
        # Start from the best unbinding query
        unbound_doc = unbind(vsa_doc, best_query['items'])
        
        # Find the greedy solution
        doc_factors, _ = solver.query(unbound_doc, target = item_encodings[-1])
        
        if len(doc_factors['features']) > 0 and doc_factors['sim'] >= best_sim[target]:
            unbound_doc = unbind(
                vsa = unbound_doc, 
                items = [torch.from_numpy(codebook.loc[feature].values).to(vsa_doc.device) for feature in doc_factors['features']])
            
            best_sim = pd.Series(
                data = torch.cosine_similarity(
                    x1 = unbound_doc, 
                    x2 = torch.from_numpy(codebook.values).to(vsa_doc.device)).cpu(),
                index = codebook.index).sort_values(ascending=False)
            
            best_query['names'].extend(doc_factors['features'])
            best_query['operations'] +=  ' ⊙ (' + ' ⊙ '.join(doc_factors['features']) + ')'
            best_query['type'] = 'cleaned_' + best_query['type']           
        
    # If the best similarity is too low, restore the original encoding
    if best_sim.max().item() < 0.1 and best_sim.max().item() != best_sim[target].item() and best_query['type'] != 'original':
        best_query = queries[-1]
        best_sim = pd.Series(
            data = torch.cosine_similarity(
                x1 = vsa_doc, 
                x2 = torch.from_numpy(codebook.values).to(vsa_doc.device)).cpu(),
            index = codebook.index).sort_values(ascending=False)
        
    # Find the meaning of the best matching item
    best_item = best_sim.index[0]
    matching_types = [query['type'] for query in queries if best_item in query['names'] and len(query['names']) == 1 and query['type'] not in ['greedy', 'original']]
    if best_item == target:
        best_item_type = 'target'
    elif matching_types:
        best_item_type = '|'.join(matching_types) 
    elif best_item in pairs.keys() and (
        best_item in pairs[best_item] or 
        any(best_item in pairs.get(item, []) for item in best_query['names'])):
        best_item_type = 'pair_values'
    elif any([(best_item in w) or (w in best_item) for w in words]):
        best_item_type = 'related'
    else:
        best_item_type = next((q['type'] for q in reversed(queries) if best_item in q['names']), 'out-of-context')
        
    # Create the set of factors based on the query and extracted factor
    extracted_factors = [best_item_type] if best_sim.max() >= 0.1 else []
    if best_query['type'] not in ['original', 'greedy']:
        extracted_factors.append(best_query['type'])
    if best_query['type'] == 'greedy':
        
        # Get the used features
        feature = best_query['names'][0]
        
        # Check if the feature is in the queries (it should not be)
        query_type = next((q['type'] for q in reversed(queries) if feature in q['names']), None)
 
        # Check if the feature is in the pairs
        if query_type:
            factor_type = query_type
        elif feature in pairs.keys() and (feature in pairs.get(best_item, []) or any(feature in pairs.get(item, []) for item in best_query['names'])):
            factor_type = 'pair_values'
        elif any([(feature in w) or (w in feature) for w in words]):
            factor_type = 'related'
        else:
            factor_type = 'out-of-context'
        extracted_factors.insert(0, factor_type)

    if 'example_key' in extracted_factors and 'example_value' in extracted_factors:
        extracted_factors.append('example')
        extracted_factors.remove('example_key')
        extracted_factors.remove('example_value')
    extracted_factors = sorted(extracted_factors)
    
    # Compute the precision
    precision_1 = best_item == target
    precision_3 = target in best_sim.index[:3]
    precision_5 = target in best_sim.index[:5] 
    
    # Unrelated baseline
    #precision_1 = best_item in words
    #precision_5 = any([w in best_sim.index[:5] for w in words])
    
    if verbose:
        print(f'\nITEMS ({len(item_encodings)}):', ' | '.join(items))
        print( '-' * 50, 
              f'\nDOC: "{partial_doc}"', 
              f'\nNEXT TOKEN: {next_token}', 
              f'\nTARGET RANK: {target_rank}',
              f'\nUNBINDING COMBO ({best_query["type"]}):', best_query['operations'], 
              '\nEXTRACTED FACTORS: ' + ' | '.join(extracted_factors),
              f'\nSIM TARGET: {target}:', round(best_sim[target].item(),2) if target in best_sim.index else 0)
        print('-' * 50)
        print('-' * 9, "Codebook's cosine similarities", '-' * 9)
        print('-' * 50)
        print(best_sim.round(2))
        print('-' * 50)

    # Save the results
    results = {
        'doc': partial_doc, 
        'target': target,
        'next_token': next_token, 
        'target_token_rank': target_rank,
        'unbunding_combo': best_query['operations'],
        'unbunding_type': best_query['type'],
        'best_item_type': best_item_type,
        'extracted_factors': '|'.join(extracted_factors),
        'target_vsa_cosine_sim': round(best_sim[target].item(), 2) if target in best_sim.index else 0,
        'vsa_sim': {label: round(best_sim, 2) for label, best_sim in best_sim[best_sim >= 0.1].iloc[:5].items()},
        'precisions': dict(zip(['precision@1', 'precision@3', 'precision@5'], [precision_1, precision_3, precision_5]))
    }
    
    return results