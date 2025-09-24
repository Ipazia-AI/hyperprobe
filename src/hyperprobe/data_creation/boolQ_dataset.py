
from os import path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# LOCAL IMPORTS
from hyperprobe.data_creation.utils import linguistic_utils

def create_augmented_questions(df):
    
    items = []
    for _, item in df.iterrows():
        
        # Extract and sort the features based on their position in the document
        mapped_features = {feature.split('-')[0].lower().replace('_', ' ').strip() : feature for feature in item['features']}
        features = list(mapped_features.keys())
        features = sorted(features, key=lambda x: item['doc'].find(x))
        
        # Create augmented questions by progressively adding features
        for idk_feature, f in enumerate(features):
            start_pos = item['doc'].find(f)
            end_pos = start_pos + len(f) + 1
            
            # Extract the partial document up to and including the current feature
            partial_doc = item['doc'][:end_pos].strip()
            doc_features = [mapped_features[f] for f in features[:idk_feature + 1]]
        
            # Handle edge case where the partial document is the same as the original document
            if partial_doc.strip('?') == item['doc'].strip('?'):
                partial_doc = item['doc'][start_pos:end_pos].strip(' ?')
                doc_features = [mapped_features[f]]
            
            # Append the new item
            items.append({'doc': partial_doc, 'features': doc_features})
            
    # Convert the items to a DataFrame
    df = pd.DataFrame(items)
        
    return df

if __name__ == "__main__":
    
    # Load the BoolQ dataset
    ds = load_dataset("google/boolq")
    
    # Convert the train and validation splits to a single DataFrame
    df = pd.concat([ds['train'].to_pandas() , ds['validation'].to_pandas()], ignore_index=True)
    
    # Add a question mark to the end of each question and capitalize it
    df['question'] = df['question'].str.capitalize()
    
    # Extract the 'question' column as a list
    docs = df['question'].tolist() # .sample(10)

    # Analize the docs
    nlp = linguistic_utils.load_spacy_model()
    outputs = []
    for doc in tqdm(nlp.pipe(docs), desc='Linguistic analysis', total=len(docs)):
        features = linguistic_utils.linguistic_features(doc, verbose=False)
        
        # Add a question mark to the end of the doc if it is a question
        text = doc.text.strip()
        if doc[0].pos_ ==  'AUX':
            text += '?'
        
        # If features are found, append them to the outputs
        if len(features) > 0:
            outputs.append({'doc': text, 'features': features})
            
    # Convert the outputs to a DataFrame and merge with the original DataFrame
    outputs = pd.DataFrame(outputs) 
    outputs['doc2'] = outputs['doc'].str.rstrip('?')
    outputs = outputs.merge(df.set_index('question')['answer'], how='left', left_on='doc2', right_index=True).drop(columns=['doc2'])
    
    print(outputs)
    
    # Create augmented questions for the training set
    augmented_outputs = create_augmented_questions(outputs)
    
    # Save the outputs to as a Json file
    output_folder = 'data'
    outputs.to_json(path.join(output_folder, 'questions.json'), orient='records', force_ascii=False, indent=4)
    augmented_outputs.to_json(path.join(output_folder, 'training_questions.json'), orient='records', force_ascii=False, indent=4)
    