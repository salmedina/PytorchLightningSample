import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics import Accuracy
from models import ResNet
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from rich.pretty import pprint


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
    
    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform)
            self.train_data, self.val_data = random_split(self.train_set, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_data = datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=12)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=12)
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, num_workers=12)
        return test_loader


class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = ResNet()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy()

        self.config = config
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        acc = self.accuracy_metric(logits, y)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss}

    def training_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_metric.compute())
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        # Need last error value and save_path from config
        pass


def load_config(path):
    yaml = YAML(typ='safe')
    return yaml.load(open(path))


def inject_sweep_to_config(sweep_config, config, separator='.'):
    for key, value in sweep_config.items():
        key_tokens = key.split(separator)
        pin = config
        for key_level in range(len(key_tokens) - 1):
            pin = pin[key_tokens[key_level]]
        prev_value = pin[key_tokens[-1]]
        pin[key_tokens[-1]] = value
        print(f'Updated {key}: {prev_value} -> {value}')


def train_and_test():
    config = edict(lr=1e-1, batch_size=32)
    wandb.init(wandb.init(project='w2vc_riglogic',
                          entity='epic-games-ml-team-1',
                          group='pl-tutorial',))
    wandb_logger = WandbLogger(project='w2vc_riglogic',
                               entity='epic-games-ml-team-1',
                               group='pl-tutorial',
                               config=config)

    mnist_data = MNISTDataModule(batch_size=config.batch_size)
    img_clf = ImageClassifier(config=config)
    
    trainer = pl.Trainer(max_epochs=10,
                         accelerator='gpu', devices=1,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         enable_checkpointing=False)
    
    # Train
    trainer.fit(img_clf, mnist_data)

    # Test
    trainer.test(img_clf, mnist_data)


def sweep_hyperparams_iter():
    # Helper function for sweeps
    wandb.init(group=wandb.config.group)
    
    wandb_logger = WandbLogger(config=wandb.config)

    mnist_data = MNISTDataModule(batch_size=wandb.config.batch_size)
    img_clf = ImageClassifier(wandb.config)

    trainer = pl.Trainer(logger=wandb_logger,
                         accelerator='gpu', devices=1,
                         max_epochs=5)
    trainer.fit(img_clf, mnist_data)


def sweep_hyperparams(config):
        
    sweep_config = load_config('config\sweeps\sample.yaml')
    sweep_id = wandb.sweep(sweep_config,
                           entity=config.wandb.entity,
                           project=config.wandb.entity)

    wandb.agent(sweep_id, function=sweep_hyperparams_iter)


def sweep_print_iter():
    wandb.init(group=wandb.config.group)
    
    sweep_config = load_config('config/sweeps/sample.yaml')
    sweep_id = wandb.sweep(sweep_config,
                           entity='epic-games-ml-team-1',
                           project='w2vc_riglogic')

    pprint(wandb.config)


def sweep_print(config):
    sweep_config = load_config('config/sweeps/sample.yaml')
    sweep_id = wandb.sweep(sweep_config,
                           entity=config.wandb.entity,
                           project=config.wandb.project)

    wandb.agent(sweep_id, function=sweep_print_iter)


if __name__ == '__main__':
    config_path = 'config/train/sample_config.yaml'
    config = load_config(config_path)

    # train_and_test(config)
    # sweep_hyperparams(config)
    sweep_print(config)
