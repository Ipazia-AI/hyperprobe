from lightning import LightningDataModule, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor, GradientAccumulationScheduler
from datetime import datetime
from os import makedirs, path
import json
import torch


# LOCAL IMPORTS
from hyperprobe.encoder.utils.encoder import VSAEncoder

def train_hyperprobe(loader:LightningDataModule, configs:dict = None, device:torch.device = None):
    
    # Random seed for reproducibility
    torch.manual_seed(101)

    # Load the neural VSA encoder
    model = VSAEncoder(
        input_dim = loader.get_input_dim(), 
        output_dim = loader.get_target_dim(),
        learning_rate = 1e-4, 
        weight_decay = 1e-4, 
        drop_p = 0.5)
    
    # Initialize the logger
    logger = TensorBoardLogger(
        save_dir = path.join(configs['app']['folder'], '_logs'), 
        name = datetime.now().strftime("%d%b").lower(),
        version = configs['app']['run_name']+ '_' + model.model.name,
        sub_dir = datetime.now().strftime("%Hh%Mm"), 
        default_hp_metric = False
    )

    # Save the model hyperparameters
    makedirs(logger.log_dir, exist_ok=True)
    with open(path.join(logger.log_dir, 'app_configs.json'), 'w') as f:
        json.dump(configs, f, indent=4)
    with open(path.join(logger.log_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model.model))

    # Clear the GPU cache
    torch.cuda.empty_cache()
    
    # TRAINING: Define the model and the trainer
    trainer = Trainer(
        devices = [device.index if device else torch.cuda.device_count() - 1],
        precision = 'bf16-mixed', 
        logger = logger,
        max_epochs = configs['epochs'], 
        deterministic = True,
        callbacks = [
            EarlyStopping(monitor = "val_loss", patience = 200, mode = 'min', verbose = False),
            ModelCheckpoint(
                monitor = "val_sim",
                mode = 'max', 
                save_top_k = 1,
                dirpath = path.join(configs['app']['folder'], 'models', datetime.now().strftime("%d%b").lower()), 
                filename = configs['app']['run_name'] + '_' + model.model.name + '_{val_sim:.0%}'),
            LearningRateMonitor(logging_interval='epoch'),
            GradientAccumulationScheduler(scheduling = {110: 2, 310: 4, 410: 8})
            ]
        )

    # Find the optimal learning rate
    results = Tuner(trainer).lr_find(model, datamodule=loader, num_training = 200, max_lr=1e-3)
    results.plot(suggest=True).savefig(path.join(logger.log_dir, 'lr_find.pdf'))

    # Add the SWA callback
    trainer.callbacks.append(StochasticWeightAveraging(swa_lrs = model.learning_rate * 10, swa_epoch_start = 400)) 
    
    # Training process
    trainer.fit(model, datamodule=loader)
    
    # Get the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    
    # Evaluate the model on the test set
    test_metrics = trainer.test(model = model, datamodule=loader, ckpt_path='best')[0]
    
    # Save the results
    with open(path.join(trainer.log_dir, 'results.json'), 'w') as f:
        json.dump(test_metrics, f)
        
    return best_model_path, test_metrics
        