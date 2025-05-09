from collections import defaultdict
import itertools
import re
import numpy as np
import pandas as pd

def create_numerical_examples():
    
    # Create the domains
    domains = ['math_squares', 'math_cubes', 'math_division2','math_division5', 'math_division10', 'math_double', 'math_root']
    numbers = pd.DataFrame(
        data = np.concatenate([
            np.arange(2, 100 + 1, step = 2),
            np.arange(5, 100 + 1, step = 5),
            np.arange(10, 100 + 1, step = 10)]), 
        columns = ['a']).sort_values('a').drop_duplicates().reset_index(drop = True)
    #all_numbers = numbers['a'].values.tolist()
    
    examples = defaultdict(list)
    flipped_examples = defaultdict(list)
    all_numbers = defaultdict(list)
    verbose_examples = defaultdict(list)
    for domain in domains:
        
        df = numbers.copy() 
        
        if domain == 'math_double':
            df['b'] = df['a'] * 2
        elif domain == 'math_squares':
            df['b'] = df['a'] ** 2
        elif domain == 'math_cubes':
            df['b'] = df['a'] ** 3
        elif domain == 'math_division2':
            df['b'] = df['a'] / 2
        elif domain == 'math_division5':
            df['b'] = df['a'] / 5
        elif domain == 'math_division10':
            df['b'] = df['a'] / 10
        elif domain == 'math_root':
            df['b'] = df['a'].map(lambda x: np.sqrt(x))
        
        # Filter the integers
        df['b'] = df['b'].map(lambda x: x if isinstance(x, int) else x if x.is_integer() else None)
        df = df.dropna().astype(int)
        
        # Filter the values
        df = df[df['b'] < 1000]
        #all_numbers.extend(df['b'].values.tolist())
        
        # Store the unique numbers
        for n in np.unique(df.values.flatten()).tolist():
            all_numbers[str(n)].append(domain)
        
        # All possible combinations
        random_pairs = list(itertools.permutations(df.index, 2))
        random_pairs = np.unique(random_pairs, axis = 0)
        
        # Create the examples
        template = lambda x, y : f" {x[0]} : {x[1]} = {y[0]} : {y[1]}"
        verbose_template = lambda x, y: f" {x[0]} is to {x[1]} as {y[0]} is to {y[1]}"
        for pair_a, pair_b in random_pairs:
            examples[domain].append(template(df.loc[pair_a].values, df.loc[pair_b].values))
            verbose_examples[domain].append(verbose_template(df.loc[pair_a].values, df.loc[pair_b].values))
    
    return examples, flipped_examples, verbose_examples, all_numbers

def create_verbose_examples(analogy_data, bats_data):
    
    template = lambda a1, a2, b1, b2: f" {a1} is to {a2} as {b1} is to {b2}"
    noFlipping = ['hypernyms', 'meronyms', 'name_nationality', 'name_occupation', 'animal_shelter', 'country_language', 'things_color', 'UK_city_county']
    
    # Create the examples
    docs = defaultdict(list)
    
    # Analogy data
    for domain, df in analogy_data.items():
        
        # Standard order
        docs[domain].extend(
            df.apply(lambda x: template(x['a1'], x['a2'], x['b1'], x['b2']), axis = 1).unique().tolist())
        
        # Flip the key and value
        if all([domain not in item for item in noFlipping]):
            docs[domain].extend(
                df.apply(lambda x: template(x['a2'], x['a1'], x['b2'], x['b1']), axis = 1).unique().tolist())
    
    # BATS data
    for domain, df in bats_data.items():
        
        # Generate all possible combinations
        combinations = list(itertools.combinations(df.index, 2))
        
        # Generate the texts for each combination
        for combo in combinations:
            values = df.loc[list(combo)].values.flatten()

            # Standard order
            if 'hyponyms' not in domain:
                docs[domain].append(template(*values))
            
            # Flip the key and value
            if all([domain not in item for item in noFlipping]):
                docs[domain].append(template(values[1], values[0], values[3], values[2]))
            
        # Avoid duplicates
        docs[domain] = np.unique(docs[domain]).tolist()
        
    return docs
    

def create_mixed_examples(df_a, df_b):
    
    # Concatenate the columns for the first df
    df_a = {
        domain: pd.DataFrame({
            'a': pd.concat([df['a1'], df['b1']], ignore_index=True),
            'b': pd.concat([df['a2'], df['b2']], ignore_index=True)
        }).drop_duplicates().reset_index(drop=True) 
        for domain, df in df_a.items()}
    
    all_texts = dict()
    all_texts_reversed = dict()
    for df in [df_a, df_b]:
        for domain, df in df.items():
            
            # Consider also the inverse pair
            #df = pd.concat([df, df[reversed(df.columns)]], ignore_index=True)

            # Generate the random pairs
            random_pairs = list(itertools.permutations(df.index, 2))
            random_pairs = np.unique(random_pairs, axis = 0)

            # Generate texts with different patterns
            texts = []
            reversed_texts = []
            for pair_a, pair_b in random_pairs:
                
                # Pattern 1: KEY A : KEY B = VALUE A : VALUE B
                texts.append(f" {df.loc[pair_b, 'b']} : {df.loc[pair_a, 'b']} = {df.loc[pair_b, 'a']} : {df.loc[pair_a, 'a']}")
                texts.append(f" {df.loc[pair_b, 'a']} : {df.loc[pair_a, 'a']} = {df.loc[pair_b, 'b']} : {df.loc[pair_a, 'b']}")
      
                # Pattern 2: KEY A : VALUE B = VALUE A : KEY B
                reversed_texts.append(f" {df.loc[pair_a, 'a']} : {df.loc[pair_b, 'b']} = {df.loc[pair_b, 'a']} : {df.loc[pair_a, 'b']}")
                reversed_texts.append(f" {df.loc[pair_b, 'a']} : {df.loc[pair_a, 'b']} = {df.loc[pair_a, 'a']} : {df.loc[pair_b, 'b']}")
            
            # Store the texts
            all_texts[domain] = np.unique(texts).tolist()
            all_texts_reversed[domain] = np.unique(reversed_texts).tolist()
    
    # Remove duplicated items
    print('\nGENERATED TEXTS:', sum([len(texts) for texts in all_texts.values()]), '\n')

    return all_texts, all_texts_reversed


def create_train_val_test_splits(examples, predetermined_train_data):
    
    # Extract all pairs and their corresponding documents
    all_targets = set()
    doc_to_pair = {}
    for docs in examples.values():
        for doc in docs:
            
            # Extract the tokens of the doc
            tokens = re.split(r"\s*[:=]\s*", doc.strip())
            item = tokens[-1]
            
            all_targets.add(item)
            doc_to_pair[doc] = item
        
    # Split into train, val, and test sets
    dim = {
        'train': int(np.ceil(len(all_targets) * 0.7)),
        'val': int(np.ceil(len(all_targets) * 0.15)),   
        'test': int(np.ceil(len(all_targets) * 0.15))
    }

    # Remove the predetermined train data from the unique targets
    predetermined_train_data = set([item for item in predetermined_train_data if item in all_targets])
    targets_to_assign = all_targets - predetermined_train_data
    
    # Shuffle the pairs
    np.random.default_rng(seed = 101).shuffle(list(targets_to_assign))
     
    # Initialize the train set with the items with multiple values
    splits = {'train': predetermined_train_data, 'val': set(), 'test': set()}
    
    # Assign the rest of the items to the splits
    for item in targets_to_assign:
        if len(splits['train']) <= dim['train']:
            splits['train'].add(item)
        elif len(splits['val']) <= dim['val']:
            splits['val'].add(item)
        elif len(splits['test']) <= dim['test']:
            splits['test'].add(item) 
        else:
            selected_split = [(set_name, len(items) / dim[set_name]) for set_name, items in splits.items()]

            selected_split = sorted(selected_split, key = lambda x: x[1])[0][0]
            splits[selected_split].add(item) 
            
    # Double check the splits
    common_items = set.intersection(*splits.values())
    assert len(common_items) == 0, 'ERROR: Common items found in splits'
    assert len(set.union(*splits.values())) == len(all_targets), f'ERROR: Mismatch in the number of items --> TOTAL TARGETS: {len(all_targets)} | SPLITS: {len(set.union(*splits.values()))}'
    print('ITEMS:', len(all_targets),  '-->', {set_name : round(len(items) / len(all_targets), 2) for set_name, items in splits.items()})

    # Assign each document to a split based on its pair
    splitted_data = {split: set() for split in splits.keys()}
    for doc, pair in doc_to_pair.items():
        if pair in splits['train'] or len(doc.strip().split()) == 1:
            splitted_data['train'].add(doc)
        elif pair in splits['val']:
            splitted_data['val'].add(doc)
        elif pair in splits['test']:
            splitted_data['test'].add(doc)
        else:
            print('ERROR: Pair not found in any split', doc, pair)  
    splitted_data = {split: sorted(docs) for split, docs in splitted_data.items()}         
            
    # Count the number of examples in each split
    tot_docs = sum([len(docs) for docs in splitted_data.values()])
    tot_examples = sum([len(docs) for docs in examples.values()])
    print(f'SPLIT ({tot_docs}):', {set_name: round(len(items) / tot_docs, 3)  for set_name, items in splitted_data.items()})
    
    # Check the total number of examples
    if tot_docs != tot_examples:
        print(f'--> WARNING: Removed some duplicated docs ({tot_examples - tot_docs}, {round(((tot_examples - tot_docs) / tot_examples) * 100, 1)} %).\n')
    
    return splitted_data

def extract_unique_pairs(examples):
    """Extract unique pairs from each document in examples."""
    unique_pairs = defaultdict(set)
    for docs in examples.values():
        for doc in docs:
            
            # Split using both separators and check we have at least four parts
            words = re.split(r"\s*[:=]\s*", doc.strip())
            pair_a, pair_b = words[:2], words[2:]
            
            # Add the pair A to the dictionary
            unique_pairs[pair_a[0]].add(pair_a[1])
            unique_pairs[pair_a[1]].add(pair_a[0])
            
            # Add the pair B to the dictionary
            unique_pairs[pair_b[0]].add(pair_b[1])
            unique_pairs[pair_b[1]].add(pair_b[0])
    
    # Convert the sets to lists
    unique_pairs = {key: sorted(value) for key, value in unique_pairs.items()}        
    
    return unique_pairs


def generate_randomExamples(items):
    
    template = lambda a1, a2, b1, b2: f" {a1} is to {a2} as {b1} is to {b2}"
    
    # Generate all possible permutations of the items
    samples = np.array([np.random.choice(list(items.keys()), size=4, replace=False) for _ in range(10000)])
    
    # Create the random examples
    docs = {'random': [template(*sample) for sample in samples]}

    return docs
