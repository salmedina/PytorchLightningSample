import pytorch_lightning as pl
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
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data.dir
        self.batch_size = config.data.batch_size
        self.transform = transforms.ToTensor()

        self.config = config

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = datasets.MNIST(
                self.data_dir, train=True, download=False, transform=self.transform)
            self.train_data, self.val_data = random_split(
                self.train_set, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_data = datasets.MNIST(
                self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data,
                                  batch_size=self.batch_size,
                                  num_workers=self.config.data.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,
                                batch_size=self.batch_size,
                                num_workers=self.config.data.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_data,
                                 batch_size=self.batch_size,
                                 num_workers=self.config.data.num_workers)
        return test_loader


class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = ResNet()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=10)

        self.config = config

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)

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


def train_and_test(config):
    wandb.init(project=config.wandb.project,
               entity=config.wandb.entity,
               group=config.wandb.group,
               config=config)

    wandb_logger = WandbLogger(project=config.wandb.project,
                               entity=config.wandb.entity,
                               group=config.wandb.group,
                               config=config)

    mnist_data = MNISTDataModule(config)
    img_clf = ImageClassifier(config)

    trainer = pl.Trainer(max_epochs=config.training.num_epochs,
                         accelerator='gpu', devices=1,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         enable_checkpointing=False)

    # Train
    trainer.fit(img_clf, mnist_data)

    # Test
    trainer.test(img_clf, mnist_data)


def sweep_hyperparams(sweep_config, config):
    '''Sweep hyperparameters'''

    def sweep_hyperparams_iter():
        '''Helper function to iterate over sweep hyperparameters'''

        wandb.init(group=config.wandb.group)

        inject_sweep_to_config(wandb.config, config, separator='.')

        wandb_logger = WandbLogger(config=config)

        mnist_data = MNISTDataModule(config)
        img_clf = ImageClassifier(config)

        trainer = pl.Trainer(logger=wandb_logger,
                             accelerator='gpu', devices=1,
                             max_epochs=config.training.num_epochs)
        trainer.fit(img_clf, mnist_data)

    # Start of Sweep
    wandb.login()
    sweep_id = wandb.sweep(sweep_config,
                           project=config.wandb.project,
                           entity=config.wandb.entity)

    wandb.agent(sweep_id, function=sweep_hyperparams_iter)


if __name__ == '__main__':
    sweep_config_path = 'config/sweeps/sample.yaml'
    sweep_config = load_config(sweep_config_path)

    base_config_path = 'config/train/base.yaml'
    base_config = edict(load_config(base_config_path))

    sweep_hyperparams(sweep_config, base_config)
    # train_and_test(base_config)
