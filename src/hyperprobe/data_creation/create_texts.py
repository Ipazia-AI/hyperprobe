from os import path
import json
import pandas as pd

# LOCAL IMPORTS
from hyperprobe.data_creation.utils import analogies as utils
from hyperprobe.data_creation.utils import data_utils

if __name__ == '__main__':
    
    root_folder = 'data'

    # Define the separators
    separators = {
        'inner': ' : ',
        'example': ' = '
    }
    
    # Load the data
    analogy_data = utils.import_analogies_file(root_folder)
    bats_data, items_with_multiple_values = utils.import_bats_files(folder = path.join(root_folder, 'BATS_3.0'))
    
    # Remove the city_in_state dataset --> multiple possible answers
    analogy_data.pop('city_in_state')
    for item in ['animal_young', 'things_color', 'synonyms_intensity', 'synonyms_exact', 'meronyms_part','meronyms_member', 'meronyms_substance']:
        bats_data.pop(item)
    
    # Uniformize the bats dataset
    analogy_data, bats_data = utils.uniformise_texts(analogy_data, bats_data)
    
    # Extract unique features
    unique_items = utils.extract_unique_features(analogy_data)
    unique_items_bat = utils.extract_unique_features(bats_data)
    unique_items.update(unique_items_bat)
    unique_items = dict(sorted(unique_items.items(), key = lambda x: x[0]))

    # Create the textual examples
    texts = utils.create_textual_examples(analogy_data, inner_separator = separators['inner'], example_separator = separators['example'])
    flipped_texts = utils.create_textual_examples(analogy_data, inner_separator = separators['inner'], example_separator = separators['example'], flip = True)

    # Create the textual examples from the bats dataset
    texts_bats, _ = utils.create_textual_examples_bats(bats_data, inner_separator = separators['inner'], example_separator = separators['example'])
    flipped_texts_bats, ambiguous_items = utils.create_textual_examples_bats(bats_data, inner_separator = separators['inner'], example_separator = separators['example'], flip = True)
        
    # Create numerica examples
    numericals, flipped_numericals, verbose_numericals, all_numbers = data_utils.create_numerical_examples()
    unique_items.update(all_numbers)
    
    # Augment the unique features
    #random_examples = data_utils.generate_randomExamples(unique_items)

    # Merge the two datasets
    examples = texts | texts_bats | numericals
    flipped_texts = flipped_texts | flipped_texts_bats | flipped_numericals

    # Save the pairs 
    unique_pairs = data_utils.extract_unique_pairs(examples)
    
    # Create examples by mixing the domains
    mixed_examples, reversed_mixed_examples = data_utils.create_mixed_examples(analogy_data, bats_data)
    
    # Aggregate all examples
    all_examples = examples
    for domain, items in examples.items():
        if domain in mixed_examples:
            all_examples[domain].extend(mixed_examples[domain])
        if domain in flipped_texts:
            all_examples[domain].extend(flipped_texts[domain])
            
    # Create verbose examples
    verbose_examples = data_utils.create_verbose_examples(analogy_data, bats_data)
    verbose_examples.update(verbose_numericals)
    
    # Split the data
    items_with_multiple_domains = [item for item, domains in unique_items.items() if len(domains) > 1 and not item.isnumeric()]
    predetermined_train_data = items_with_multiple_values + ambiguous_items + items_with_multiple_domains
    splitted_data = data_utils.create_train_val_test_splits(all_examples, predetermined_train_data)
    
    # Check the splits by domains
    domains_per_split = {split: set([domain for doc in docs for domain in unique_items[doc.split()[-1]]]) for split, docs in splitted_data.items()}
    common_domains = sorted(set.intersection(*domains_per_split.values()))
    print('DOMAINS per split: ', ' | '.join([f'{split} ({len(domains)})' for split, domains in domains_per_split.items()])) 
    print(f'--> COMMON {len(common_domains)}', common_domains[:5], '\n')

    # Save the textual examples
    #with open(path.join(root_folder, 'texts.json'), 'w', encoding='utf-8') as file:
    #    json.dump(examples, file, indent=4, ensure_ascii=False)
        
    #with open(path.join(root_folder, 'flipped_texts.json'), 'w', encoding='utf-8') as file:
    #    json.dump(flipped_texts, file, indent=4, ensure_ascii=False)
        
    with open(path.join(root_folder, 'splitted_data.json'), 'w', encoding='utf-8') as file:
        json.dump(splitted_data, file, indent=4, ensure_ascii=False)
        
    #with open(path.join(root_folder, 'mixed_texts.json'), 'w', encoding='utf-8') as file:
    #    json.dump(mixed_examples, file, indent=4, ensure_ascii=False)
        
    with open(path.join(root_folder, 'pairs.json'), 'w', encoding='utf-8') as file:
        json.dump(unique_pairs, file, indent=4, ensure_ascii=False)
        
    with open(path.join(root_folder, 'verbose_examples.json'), 'w', encoding='utf-8') as file:
        json.dump(verbose_examples, file, indent=4, ensure_ascii=False)
        
    # Save the unique features
    with open(path.join(root_folder, 'features.json'), 'w', encoding='utf-8') as file:
        json.dump(unique_items, file, indent=4, ensure_ascii=False)
        
    # Save the unique features
    #with open(path.join(root_folder, 'random_examples.json'), 'w', encoding='utf-8') as file:
    #    json.dump(random_examples, file, indent=4, ensure_ascii=False)
        
    # Print the stats
    stats = pd.Series({key: len(df) for key, df in examples.items()}).sort_values(ascending = False)
    print( '-' * 50 , '\nTOTAL FEATURES:', len(unique_items), '|| DOMAINS:', len(stats),'\n' + '-' * 50)
    print(stats, '\n' + '-' * 50)
    print("TOTAL EXAMPLES:", stats.sum(),'\n' + '-' * 50)