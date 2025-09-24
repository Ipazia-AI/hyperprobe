from os import path, makedirs
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import torch

# LOCAL IMPORTS
from hyperprobe.probing.utils import utils
    
if __name__ == '__main__':
    
    # Set the device
    device = torch.device('cuda:' + str(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('DEVICE:', torch.cuda.get_device_name(device))
        
    # Load verbose data
    with open(path.join('data', 'squad', 'squad_test.json'), 'r') as file:
        QA_inputs = json.load(file)
    QA_inputs = np.random.default_rng(seed = 101).choice(QA_inputs, size = 10000, replace=False)

    # Load the VSA translator
    modelName = ("18sep", "llama3_QA_bundle_equal2_val_sim=44%" + '.ckpt')
    vsa_encoder = utils.load_vsaEncoder(model_name = path.join(*modelName), device=device)
        
    # Load the codebook
    codebook = pd.read_parquet(path.join('outputs', 'codebooks', 'features.parquet'))
    
    # Load the llm
    models = ['meta-llama/Llama-3.1-8B']
    llm = utils.load_llm(model_name = models[0], device = device)
    
    # Iterate over the inputs
    outputs = []
    for idk, item in enumerate(tqdm(QA_inputs)):
        outputs.append(utils.probe_QA_doc(item, codebook, llm, vsa_encoder, verbose = False))
        
    # Create the output folder
    output_folder = path.join('outputs', 'probing')
    makedirs(output_folder, exist_ok = True)

    # Save the outputs
    #today = datetime.now().strftime("%d%b").lower()
    file_path = path.join(output_folder, f"QA_{'_'.join(modelName[1].split('_')[:-2])}.json") # {today}
    with open(file_path, mode = 'w', encoding='utf-8') as file:
        json.dump(outputs, file, indent = 4, ensure_ascii=False)