from Vit import SimpleViT
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from init import WeightInitCallback
import math
import torch
import pytorch_lightning as pl
from Dataset import SubwaySurfers
from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir,train_transform=None, val_transform=None,batch_size=32,num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = SubwaySurfers(self.data_dir,transform=train_transform)
        self.val_dataset = SubwaySurfers(self.data_dir,transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch   # x shape: (batch, channels, 3, height, width)  y shape: (batch,)
        y_hat = self(x)  # y_hat shape: (batch, num_classes)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':


    data_dir = ''
    batch_size = 32
    num_workers = 4
    attention_init = "xavier_uniform"
    mlp_init = "kaiming_uniform"
    attention_gain = 1.3
    mlp_gain = math.sqrt(2.0)
    max_epochs = 100
    min_epochs = 10
    device = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed"

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    init_callback = WeightInitCallback(
        attention_init=attention_init,
        mlp_init=mlp_init,
        attention_gain=attention_gain,
        mlp_gain=mlp_gain,
        excluded_params=["bias", "positional_embedding"]
    )

    model = SimpleViT(image_size=224, image_patch_size=32, frames=3, frame_patch_size=1, num_classes=5, dim=512,
                      heads=8, mlp_dim=2048, num_transformer_layers=4, channels=3,depth=1)

    datamodule = DataModule(data_dir, train_transform, val_transform,batch_size=batch_size,num_workers=num_workers)
    model = Model(model)

    trainer = pl.Trainer(max_epochs=max_epochs,min_epochs=min_epochs,accelerator=device,devices=1,callbacks=[init_callback, early_stopping],check_val_every_n_epoch=1,
        log_every_n_steps=5,
        enable_progress_bar=True,precision=precision)

    trainer.fit(model, datamodule)


