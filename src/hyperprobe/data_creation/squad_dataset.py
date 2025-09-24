import pandas as pd
import tensorflow_datasets as tfds
from os import makedirs, path, environ
environ["CUDA_VISIBLE_DEVICES"] = ""
    
from tqdm import tqdm
from difflib import get_close_matches

# LOCAL IMPORTS
from hyperprobe.data_creation.utils import linguistic_utils

def augment_items(df):
    items = []
    for _, item in df.iterrows():
        
        # Extract and sort the features based on their position in the document
        all_features = item['question_features'] + [f for features in item['answer_features'] for f in features]
        mapped_features = {feature.split('-')[0].lower().replace('_', ' ').strip(): feature for feature in all_features}
        features = list(mapped_features.keys())
        
        #context, question = item['doc'].split('\n\n')
        doc = item['question'] + ' ' + item['answers'][0]

        # Create augmented questions by progressively adding features
        for idk_feature, f in enumerate(features):
            start_pos = doc.lower().find(f)
            
            # Backup plan if the feature is not found in the document
            if start_pos == -1:
                matches = get_close_matches(word = f, possibilities = doc.lower().split(), n = 1, cutoff=0.7) # 
                if len(matches) == 0:
                    continue
                f = matches[0]
                start_pos = doc.lower().find(f)
            end_pos = start_pos + len(f)
            
            # Extract the partial document up to and including the current feature
            partial_doc = doc[:end_pos].strip("? '")
            doc_features = [mapped_features[f] for f in features[:idk_feature + 1]]
            
            if len(partial_doc) < 3:
                continue
            
            # Append the new item
            items.append({
                'title': item['title'],
                'doc': partial_doc, # item['context'] + '\n\n' + 
                'features': doc_features
            })
            
    # Convert the items to a DataFrame
    df = pd.DataFrame(items)
        
    return df

def download_process_squad():
    
    # Download and load SQuAD 2.0
    dataset, info = tfds.load("squad/v1.1", with_info=True)

    # Concatenate the train/validation datasets
    df = pd.concat([
        tfds.as_dataframe(dataset["train"]).assign(set='train'),
        tfds.as_dataframe(dataset["validation"]).assign(set='validation')
    ])
    
    # Datawrangling: keep only relevant columns and decode bytes to strings
    df = pd.DataFrame({
        'title': df['title'].apply(lambda text: text.decode('utf-8') if isinstance(text, bytes) else text),
        'set': df['set'],
        'context': df['context'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x),
        'question': df['question'].apply(lambda text: text.decode('utf-8') if isinstance(text, bytes) else text),
        'answers': df['answers/text'].apply(lambda items: list(set([item.decode('utf-8') if isinstance(item, bytes) else item for item in items]))),
    })
    
    # Linguistic analysis of the questions to extract features
    nlp = linguistic_utils.load_spacy_model()
    lexical_semantics_l0, lexical_semantics_l1 = [], []
    for doc in tqdm(nlp.pipe(df['question'].values), desc='Linguistic analysis', total=len(df)):
        lexical_semantics_l0.append(linguistic_utils.linguistic_features(doc, lexical_semantics_level = 0, verbose=False))
        #lexical_semantics_l1.append(linguistic_utils.linguistic_features(doc, lexical_semantics_level = 1, verbose=False))
    df['question_features'] = lexical_semantics_l0
    #df['question_features_l1'] = lexical_semantics_l1
    
    # Linguistic analysis of the answers to extract features
    lexical_semantics_l0, lexical_semantics_l1 = [], []
    for item in tqdm(df['answers'].values):
        
        # Analyze the all the answers of the document
        sem0, sem1 = [], []
        for answer in nlp.pipe(item):
            sem0.append(linguistic_utils.linguistic_features(answer, lexical_semantics_level = 0, verbose=False))
            #sem1.append(linguistic_utils.linguistic_features(answer, lexical_semantics_level = 1, verbose=False))
        
        # Save the features (if no features found, save the tokenized answer)
        lexical_semantics_l0.append(sem0) if any(len(f) > 0 for f in sem0) else lexical_semantics_l0.append([answer.replace('-', '_').split() for answer in item])
        #lexical_semantics_l1.append(sem1) if any(len(f) > 0 for f in sem1) else lexical_semantics_l1.append([answer.replace('-', '_').split() for answer in item])
        
    df['answer_features'] = lexical_semantics_l0
    #df['answer_features_l1'] = lexical_semantics_l1
    
    return df

if __name__ == "__main__":
    
    output_folder = path.join('data', 'squad')
    file_path = path.join(output_folder, 'squad_dataset.json')
    
    if not path.exists(file_path):
        df = download_process_squad()

        # Save to json
        makedirs(output_folder, exist_ok=True)
        df.to_json(path.join(output_folder, "squad_dataset.json"), orient='records', force_ascii=False, indent=4)
        print(f"SQuAD dataset with {len(df)} samples saved to {output_folder}/squad_dataset.json")
    
    df = pd.read_json(file_path)
    df['question'] = df['question'].apply(lambda text: text[0].upper() + text[1:] if len(text) > 0 else text)
    
    # Create augmented training data by progressively adding features to the questions
    training_df = augment_items(df)
    training_df.to_json(path.join(output_folder, "squad_training.json"), orient='records', force_ascii=False, indent=4)
    
    # Create the test/validation set with the full questions
    # validation_df = df.loc[df['set'] == 'validation', ['title', 'context', 'question', 'answers', 'question_features', 'answer_features']].copy()
    validation_df = df[['title', 'context', 'question', 'answers', 'question_features', 'answer_features']].copy()
    validation_df.insert(1, 'doc', 
                         value = validation_df['context'] + '\nQ: ' + validation_df['question'] + '\nA (≤ 3 words):')
    validation_df.drop(columns = ['context', 'question']).to_json(path.join(output_folder, "squad_test.json"), orient='records', force_ascii=False, indent=4)
    
    print(validation_df)