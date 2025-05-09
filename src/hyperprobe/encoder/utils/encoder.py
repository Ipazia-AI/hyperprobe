import lightning as L
import torch
import torchmetrics
import torchmetrics.classification

# LOCAL IMPORTS
from hyperprobe.encoder.utils.network import NetworkModel

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, eta_min = 0, T_mult = 1, decay_factor = 0.9, last_epoch = -1):
        self.decay_factor = decay_factor         
        super().__init__(optimizer, T_0 = T_0, T_mult = T_mult, eta_min = eta_min, last_epoch = last_epoch)
            
    def get_lr(self):
        
        # Efficient decay handling - only modify base_lrs when needed
        if self.T_cur == 0 and self.last_epoch > 0:
            self.base_lrs = [lr * self.decay_factor for lr in self.base_lrs]
        
        # Use parent's optimized cosine calculation
        return super().get_lr()
    

class VSAEncoder(L.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate, weight_decay, drop_p):
        super().__init__()
        
        # Hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        
        # Save model hyperparameters
        self.save_hyperparameters()
        
        # Initialize the model
        self.model = NetworkModel(input_dim = self.input_dim, output_dim = self.output_dim, drop_p = self.drop_p)

        # Set the precision for the matrix multiplication
        torch.set_float32_matmul_precision('high') # 'medium' | 'high'
        
        # Loss functions with weights
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.loss_weights = {"bce": 1.0, "mse": 0.1}

        # [Similarity metrics] Cosine Similarity
        self.train_sim = torchmetrics.CosineSimilarity(reduction = 'mean')
        self.val_sim = torchmetrics.CosineSimilarity(reduction = 'mean')
        self.test_sim = torchmetrics.CosineSimilarity(reduction = 'mean')

        # [Similarity metrics] Accuracy Similarity
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy() # multidim_average = 'samplewise'

    
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        
        # Print the model
        print(self.model)
        
        # Call the superclass method correctly
        super().on_train_start()
        
        # Compute the model size
        self.model_size = sum(p.numel() for p in self.parameters() if p.requires_grad) // 1e6
        
        # Log the model size
        self.log('model_size', self.model_size, on_step=False, on_epoch=True, prog_bar=False)  

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer = optimizer,
            T_0 = 100, 
            T_mult=2, 
            decay_factor=0.9, 
            eta_min = self.learning_rate * 1e-3)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _step(self, batch, batch_idx):
       
        # Unpack the batch
        x, vsa_target = batch
        
        # Forward pass
        predictions = self(x)
        
        # Combined loss
        bce_loss = self.bce_loss(predictions, normalize(vsa_target))
        mse_loss = self.mse_loss(predictions, vsa_target)
        loss = self.loss_weights["bce"] * bce_loss + self.loss_weights["mse"] * mse_loss

        return loss, vsa_target, predictions.detach()
    
    def training_step(self, batch, batch_idx):
        
        # Compute the loss, cosine similarity, and dot similarity
        loss, target, prediction = self._step(batch, batch_idx)

        # Compute similarity metrics
        #self.train_sim.update(prediction, target)
        
        # Log the metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('train_sim', self.train_sim, on_step=False, on_epoch=True, prog_bar=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, target, prediction = self._step(batch, batch_idx)

        # Compute the cosine similarity
        self.val_sim.update(prediction, target)

        # Log the metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_sim', self.val_sim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            _, target, prediction = self._step(batch, batch_idx)

        # Compute the cosine similarity
        self.test_sim.update(prediction, target)
        self.test_accuracy.update(torch.sigmoid(prediction), normalize(target))
        
        # Log the metrics
        self.log('test_sim', self.test_sim, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, sync_dist=True)
    
def normalize(x):
    return (x + 1) / 2