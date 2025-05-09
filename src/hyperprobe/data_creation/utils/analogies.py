from os import path, walk
from collections import defaultdict
from itertools import product
from transformers import AutoTokenizer
import re
import numpy as np
import pandas as pd

def import_analogies_file(root_folder):
    
    # Load the datasets
    with open(path.join(root_folder, 'analogies.txt'), 'r') as file:
        lines = file.readlines()

    # Initialize variables
    relations = defaultdict(list)
    current_relation = None

    # Process each line in the file
    for line in lines:
        line = line.strip()

        if line.startswith(':'):
            current_relation = clean_domain_name(line.lstrip(':').strip().replace('-', '_'))
        else:
            if current_relation is not None:
                relations[current_relation].append(line.split())
     
    # Turn the data into a DataFrame
    relations = {key: pd.DataFrame(value, columns=['a1', 'a2', 'b1', 'b2']).map(str.strip) # .map(str.capitalize)
                 for key, value in relations.items()}
    
    return relations


def import_bats_files(folder):
    
    file_paths = []
    for root, _, files in walk(folder):
        for file in files:
            if file != 'metadata.json':
                file_paths.append(path.join(root, file))

    # read the files
    data = defaultdict(list)
    items_with_multiple_values = set()
    for file_path in file_paths:
        
        # Initialize variables
        relation_name = clean_domain_name(path.basename(file_path).split('[')[1].split(']')[0].replace(' - ', '_'))
        
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Process the lines and save the data as a DataFrame
        df = pd.DataFrame([line.strip().split('\t') for line in lines], columns=['a', 'b'])#.map(str.capitalize)
        
        # Check if the relation has multiple values
        muliple_values = df[df['b'].apply(lambda x: len(x.split('/')) > 1)].index
        if len(muliple_values):
   
            # TRAINING: Keep the first one
            df['b'] = df['b'].apply(lambda x: x.split('/')[0].strip())
            
            # Store the pairs with multiple values to avoid testing them
            noProblem = ['antonyms_gradable', 'hypernyms_animals', 'hypernyms_misc', 'over+adj_reg']
            if len(df.loc[muliple_values]) != len(df) and (relation_name not in noProblem):
                items_with_multiple_values.update(df.loc[muliple_values, 'b'].to_list())
        
        # Strip the whitespaces
        df = df.map(str.strip)

        # Save the data
        if len(df) > 0:
            data[relation_name] = df
        
    return data, list(items_with_multiple_values)

def create_textual_examples(data,  inner_separator = ':' , example_separator = '; ', flip = False):
    
    # Define the text template
    if flip:
        text_template = lambda item: ' ' + item['a2'] + inner_separator + item['a1'] + example_separator + item['b2'] + inner_separator + item['b1']
    else:
        text_template = lambda item: ' ' + item['a1'] + inner_separator + item['a2'] + example_separator + item['b1'] + inner_separator + item['b2']

    # Create the textual examples
    texts = defaultdict(list)
    for relationName, df in data.items():
        texts[relationName] = df.apply(func = text_template, axis = 1).unique().tolist()
    return texts

def create_textual_examples_bats(data, inner_separator = ':' , example_separator = '; ', flip = False):
    
    # Define the text template
    if flip:
        text_template = lambda item: item['b'] + inner_separator + item['a']
    else:   
        text_template = lambda item: item['a'] + inner_separator + item['b']
        
    noFlipping = ['hypernyms', 'meronyms', 'name_nationality', 'name_occupation', 'animal_shelter', 'country_language', 'things_color', 'UK_city_county']
    only_flipping = ['hyponyms_misc']
    
    # Create the textual examples
    texts = dict()
    ambigius_items = set()
    for relationName, df in data.items():
        
        # Skip the domain in case of not flipping
        if not flip and relationName in only_flipping:
            continue
        
        # Create the pairs
        pair_texts = df.apply(func = text_template, axis = 1).tolist()

        # Create the examples
        docs = [' ' + example_separator.join(items) for items in product(pair_texts, pair_texts) if items[0] != items[1]]
        texts[relationName] = list(set(docs))

        if flip and any([item in relationName for item in noFlipping]):
            items = set([text.split(' : ')[-1].strip() for text in texts[relationName]])
            ambigius_items.update(items)
        
    return texts, list(ambigius_items)

def clean_text(text):
        
    # Remove parentheses
    text = re.sub(r'\(.*?\)', '', text)
    
    # Remove commas
    text = re.split(r',\s*', text)[0]
    
    # Remove prefix (List of )
    text = re.sub(r'\b(?:List of)\s+(.*?)\b', r'\1', text)
    
    # Remove the prefixes
    text = re.sub(r'\b(?:Murder of|Death of|Assassination of|Killing of|Crucifixion of|Fall of|History of)\s+', '', text)
    text = re.sub(r'\b(?:award and nominations received by|award and honors received by|honor and awards received by)\s+', '', text)
    
    # Remove the extra whitespaces
    text = text.strip()

    return text

def extract_unique_features(data):
    features = defaultdict(list)
    for domain, df in data.items():
        items = np.unique(df.values.flatten())
        for item in items:
            features[item].append(domain)
            
    # if item in features.keys():
    # print(f'WARNING ({item})', 'PREVIUS:', features[item], 'CURRENT:', domain)
            
    return features

def clean_domain_name(domain):
    to_remove = [f'gram{i}_' for i in range(10)]
    for item in to_remove:
        domain = domain.replace(item, '')
        
    domain = domain.replace('common_countries', 'world')
    return domain


def uniformise_texts(analogies, bats):
    
    # Extract the unique features
    tokezer_based = False
    if tokezer_based:
        # Load the tokenizer 
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        vocabulary = [token.strip('Ġ').strip() for token in tokenizer.get_vocab().keys()]
        
        # Drop the duplicates
        vocabulary = pd.DataFrame(vocabulary, columns = ['original'])
        vocabulary['lower'] = vocabulary['original'].str.lower()
        vocabulary['count'] = vocabulary['original'].map(vocabulary['original'].value_counts())
        vocabulary = vocabulary.sort_values(by = 'count', ascending = False)
        vocabulary = vocabulary.drop_duplicates(subset = 'lower')
        voc_mapper = {word.lower(): word for word in vocabulary['original']}
    else:
        original_words = list({word for df in analogies.values() for word in df.melt().value.unique()})
        voc_mapper = {word.lower():word for word in original_words}

    # Create the mapping function
    mapping_func = lambda word: voc_mapper[word.lower()] if word.lower().strip() in voc_mapper.keys() else word
    
    # Mapping the analogies dataset
    #for df in analogies.values():
    #    for col in df.columns:
            #df[col] = df[col].apply(mapping_func)
            
            # Check the capitalization
            #capitalization_perc = df[col].drop_duplicates().str.istitle().value_counts(normalize = True)
            #if True in capitalization_perc.index and capitalization_perc[True] >= 0.6:
            #    df[col] = df[col].str.capitalize()
            #if False in capitalization_perc.index and capitalization_perc[False] >= 0.6:
            #    df[col] = df[col].str.lower()
    
    # Mapping the bat dataset
    for df in bats.values():
        for col in df.columns:
            df[col] = df[col].apply(mapping_func)
            
            capitalization_perc = df[col].drop_duplicates().str.istitle().value_counts(normalize = True)
            if True in capitalization_perc.index and capitalization_perc[True] > 0.5:
                df[col] = df[col].str.capitalize()
            if False in capitalization_perc.index and capitalization_perc[False] > 0.5:
                df[col] = df[col].str.lower()
            
    return analogies, bats