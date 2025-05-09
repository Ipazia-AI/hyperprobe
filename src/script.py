# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
import hyperprobe

import json
import multiprocessing as mp
from os import makedirs, path
from tqdm import tqdm

if __name__ == "__main__":
    
    # [0] Labeled data for training 
    inputs = [
        {'doc': 'Denmark : krone = Mexico : peso', 'concepts': [('Denmark','krone'), ('Mexico', 'peso')]},
        {'doc': 'Berlin : Germany = Tokyo : Japan', 'concepts': [('Berlin', 'Germany'), ('Tokyo', 'Japan')]},
        {'doc': 'introvert : extravert = big : small', 'concepts': [('Introvert', 'Extravert'), ('Big','Small')]}
    ]
      
    # [1] Create the codebook for the VSA encodings: associated concepts to high-dimensional VSA representations
    all_concepts = set([concept for item in inputs for pair in item['concepts'] for concept in pair])
    codebook = hyperprobe.create_codebook(concepts = all_concepts, vsa_dimension=4096)
    print(f"\nCodebook created with {len(codebook)} concepts and {len(codebook.columns)} dimensions.\n")

    # [2] Get the LLM embeddings: Extract embeddings for the input documents using a pre-trained LLM and cluster them using k-means
    model_name = 'openai-community/gpt2-medium' #'meta-llama/Llama-4-Scout-17B-16E'

    # Use multiprocessing to execute the function in a sandbox, releasing the GPUs afterwards
    docs = [item['doc'] for item in inputs]
    k_clusters = 5
    with mp.get_context("spawn").Pool(1) as pool:
        llm_embeddings, *_ = pool.apply(hyperprobe.ingest_embeddings, args=(docs, model_name, k_clusters))
    
    # [2a] Apply sum pooling to the LLM embeddings: Sum pooling reduces the embeddings to a single vector per document
    llm_embeddings = {doc: embedding.sum(dim=0) for doc, embedding in llm_embeddings.items()}
    for item in inputs:
        item['embeddings'] = llm_embeddings[item['doc']]
        
    # [3] Create the VSA encodings for the input sentences: Map each document to its corresponding VSA encoding based on its concepts
    for item in inputs:
        item['vsa'] = hyperprobe.create_vsa_encodings(item, codebook, verbose=True)

    # Training configurations
    configs = {
        'splits': {'val': 0.1, 'test': 0.1},
        'epochs': 100,
        'batch_size': 32,
        'app': {'folder': 'outputs', 'run_name': model_name.split('/')[-1].replace('-', '_') + "_run1"}
    }
    
    # [SHOWCASE ONLY] Dummy input duplication to demonstrate training
    inputs = inputs * 5
    
    # Prepare the input dataset
    dataset = hyperprobe.inputDataset(inputs)
    loader = hyperprobe.llm2VSA_dataloader(dataset, batch_size = configs['batch_size'], val_size = configs['splits']['val'], test_size = configs['splits']['test'])

    # [4] Train the neural VSA encoder: Train the encoder and save the best model
    best_model_path, test_metrics = hyperprobe.train_hyperprobe(loader, configs=configs)
    print(f"Trained encoder.\n--> PATH: {best_model_path}")

    # [5] Probe the VSA encodings via unbinding operations
    # [5a] Load the trained encoder
    trained_encoder = hyperprobe.VSAEncoder.load_from_checkpoint(best_model_path)
    trained_encoder.eval()
    
    # [5b] Load the language model
    llm = hyperprobe.load_llm(model_name = model_name)
    
    # [5c] Retrieve the concept pairs
    pairs = {pair[0]: pair[1:] for item in inputs for pair in item['concepts']}
    
    # [5d] Process the documents
    outputs = []
    for input in tqdm(inputs, desc = f'Processing {len(inputs)} documents'):
        probed_doc = hyperprobe.probe_doc(input['doc'], codebook, llm, trained_encoder, pairs = pairs, verbose = True)
        outputs.append(probed_doc)
    print(f"\nProbed {len(outputs)} documents.")
        
    # [5e] Save the outputs
    output_folder = path.join(configs['app']['folder'], 'probing')
    makedirs(output_folder, exist_ok = True)
    
    with open(path.join(output_folder, 'extracted_concepts.json'), mode = 'w', encoding='utf-8') as file:
        json.dump(outputs, file, indent = 4, ensure_ascii=False)