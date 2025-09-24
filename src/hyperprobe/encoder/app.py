import pandas as pd
import torch
from os import makedirs, path

# LOCAL IMPORTS
from hyperprobe.encoder.utils import data_loader
from hyperprobe.encoder.utils.app_utils import train_hyperprobe
from hyperprobe.encoder.utils.encoder import VSAEncoder

configs = dict(
    batch_size = 32,
    epochs = 5000,
    app = {
        'folder': path.join('outputs', 'hyperprobe'),
        'run_name': 'llama3' + '_QA' + '_bundle' # _bind || _bundle
    }
) 

if __name__ == '__main__':
    
    # Create the output folder
    makedirs(configs['app']['folder'], exist_ok=True)

    # Load the codebook and thier VSA encodings
    codebook = pd.read_parquet(path.join('outputs', 'codebooks', 'features.parquet'))
    print('\nCODEBOOK:', len(codebook))

    # Load the token embeddings --> splittedData || textAndFlipped
    #inputs = data_loader.load_embeddings(root_folder = path.join('outputs', 'embeddings', 'splittedData', 'clustered_L16_L32'))
    QA_inputs = data_loader.load_embeddings(root_folder = path.join('outputs', 'embeddings', 'squadTraining', 'clustered_L16_L32'))

    # Attach information to the inputs (1. target keys, 2. dataset splits, 3. VSA encodings)
    #inputs = data_loader.add_info_inputs(inputs, codebook)
    QA_inputs = data_loader.add_info_QAinputs(QA_inputs, codebook)

    # Create the dataset
    data = data_loader.inputDataset(QA_inputs)
    loader = data_loader.llm2VSA_dataloader(
        data = data, 
        split = 'predefined',
        batch_size = configs['batch_size'])
    
    # Train the neural VSA encoder
    best_model_path, test_metrics = train_hyperprobe(loader, configs = configs) # device = torch.device('cuda:1')

    # Load the best model
    torch.cuda.empty_cache()
    trained_encoder = VSAEncoder.load_from_checkpoint(best_model_path)
    print(f"\nTrained encoder loaded successfully.\n--> PATH: {best_model_path}\n--> DEVICE: {trained_encoder.device}")