from datetime import datetime
from os import path, makedirs
from tqdm import tqdm
import pandas as pd
import json
import torch

# LOCAL IMPORTS
from hyperprobe.probing.utils import utils
from hyperprobe.probing.utils.comboSolver import comboSolver
    
if __name__ == '__main__':
    
    # Set the device
    device = torch.device('cuda:' + str(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('DEVICE:', torch.cuda.get_device_name(device))
    
    # Load the pairs
    with open(path.join('data', 'pairs.json'), 'r') as file:
        pairs = json.load(file)
    pairs = {k.lower(): v for k, v in pairs.items()}

    # Load verbose data
    with open(path.join('data', 'verbose_examples.json'), 'r') as file:
        inputs = json.load(file)
        
    print('\nINPUTS:', sum([len(docs) for docs in inputs.values()]), 
          f'(AVG: {int(sum([len(docs) for docs in inputs.values()]) / len(inputs.keys()))} docs per domain, {len(inputs.keys())} domains)\n')
    
    # Load the VSA translator
    #modelName = ("10apr", "llama3_merged_bindSuperAll_equal2_val_sim=88%" + '.ckpt')
    modelName = ("10apr", "pythia_merged_bindSuperAll_equal2_val_sim=85%" + '.ckpt')
    vsaTranslator = utils.load_vsaEncoder(model_name = path.join(*modelName), device=device)
        
    # Load the codebook
    codebook = pd.read_parquet(path.join('outputs', 'codebooks', 'features.parquet'))
    
    # Load the llm
    models = ['EleutherAI/pythia-1.4b']
    llm = utils.load_llm(model_name = models[0], device = device)
    
    # Init the solver (combinatory problem) for a greedy decoding approrch
    solver = comboSolver(domains = {f'item_{i}':codebook for i in range(1)}, batch_size = 16384, device = device)
    
    # Iterate over the inputs
    domain_outputs = {}
    for idk, (domain, docs) in enumerate(inputs.items()):
        domain_outputs[domain] = [utils.probe_doc(doc, codebook, llm, vsaTranslator, pairs, solver, verbose = True) 
                                  for doc in tqdm(docs[:2], desc = f'Processing {domain} ({idk + 1}/{len(inputs)})')]
        
    # Create the output folder
    output_folder = path.join('outputs', 'probing')
    makedirs(output_folder, exist_ok = True)

    # Save the outputs
    today = datetime.now().strftime("%d%b").lower()
    file_path = path.join(output_folder, f'{modelName[1].split("_")[0]}_{today}_verbose.json')
    with open(file_path, mode = 'w', encoding='utf-8') as file:
        json.dump(domain_outputs, file, indent = 4, ensure_ascii=False)