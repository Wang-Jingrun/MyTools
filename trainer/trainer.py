import os, sys
import numpy as np
from pathlib import Path
from pypesq import pesq
from torch.utils.data import DataLoader
from matplotlib import colors, pyplot as plt

from .dcunet import *
from .utils import *
from .dataLoader import *


class DCUNetTrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        self.init_dataloader()
        self.model = DCUnet10(self.config['n_fft'], self.config['hop_length']).to(self.device)
        self.loss_fn = wsdr_fn

        optimizer = getattr(sys.modules['torch.optim'], self.config['optimizer'])
        self.optimizer = optimizer(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        # 训练策略
        sc = self.config.get('lr_scheduler', None)
        if sc is None:
            self.scheduler, self.scheduler_graph = None, None
        else:
            scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], sc.pop('name'))
            self.scheduler = scheduler(self.optimizer, **sc)
            self.scheduler_graph = None

    def init_dataloader(self):
        TRAIN_NOISY_DIR = Path(os.path.join(self.config['dataset_path'], 'noisy_trainset_28spk_wav'))
        TRAIN_CLEAN_DIR = Path(os.path.join(self.config['dataset_path'], 'clean_trainset_28spk_wav'))

        TEST_NOISY_DIR = Path(os.path.join(self.config['dataset_path'], 'noisy_testset_wav'))
        TEST_CLEAN_DIR = Path(os.path.join(self.config['dataset_path'], 'clean_testset_wav'))

        train_noisy_files = sorted(list(TRAIN_NOISY_DIR.rglob('*.wav')))
        train_clean_files = sorted(list(TRAIN_CLEAN_DIR.rglob('*.wav')))

        test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
        test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

        test_dataset = SpeechDataset(test_noisy_files, test_clean_files, self.config['n_fft'], self.config['hop_length'])
        train_dataset = SpeechDataset(train_noisy_files, train_clean_files, self.config['n_fft'], self.config['hop_length'])

        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train_epoch(self):
        self.model.train()
        train_ep_loss = 0.
        counter = 0
        for noisy_x, clean_x in self.train_loader:
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

            # zero  gradients
            self.model.zero_grad()

            # get the output from the model
            pred_x = self.model(noisy_x)

            # calculate loss
            loss = self.loss_fn(noisy_x, pred_x, clean_x, n_fft=self.config['n_fft'], hop_length=self.config['hop_length'])
            loss.backward()
            self.optimizer.step()

            train_ep_loss += loss.item()
            counter += 1

        clear_cache()
        return train_ep_loss / counter

    def test_epoch(self):
        self.model.eval()
        test_ep_loss = 0.
        counter = 0.
        for noisy_x, clean_x in self.test_loader:
            # get the output from the model
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)
            pred_x = self.model(noisy_x)

            # calculate loss
            loss = self.loss_fn(noisy_x, pred_x, clean_x, n_fft=self.config['n_fft'], hop_length=self.config['hop_length'])
            test_ep_loss += loss.item()

            counter += 1

        clear_cache()
        return test_ep_loss / counter

    def train(self):
        """
        To understand whether the network is being trained or not, we will output a train and test loss.
        """
        self.train_losses = []
        self.test_losses = []

        for e in range(self.config['epochs']):

            # first evaluating for comparison
            if e == 0:
                with torch.no_grad():
                    test_loss = self.test_epoch()

                self.test_losses.append(test_loss)
                print("Loss before training:{:.6f}".format(test_loss))

            train_loss = self.train_epoch()
            self.scheduler.step()  # update lr
            with torch.no_grad():
                test_loss = self.test_epoch()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            clear_cache()

            print("Epoch: {}/{}...".format(e + 1, self.config['epochs']),
                  "Loss: {:.6f}...".format(train_loss),
                  "Test Loss: {:.6f}".format(test_loss))

        """Visualization"""
        plt.grid()
        plt.plot(self.test_losses, label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def pesq_score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.test_pesq = 0.
        counter = 0.

        for noisy_x, clean_x in self.test_loader:
            # get the output from the model
            noisy_x = noisy_x.to(self.device)
            with torch.no_grad():
                pred_x = self.model(noisy_x)
            clean_x = torch.squeeze(clean_x, 1)
            clean_x = torch.istft(clean_x, n_fft=self.config['n_fft'], hop_length=self.config['hop_length'], normalized=True)

            psq = 0.
            for i in range(len(clean_x)):
                clean_x_16 = torchaudio.transforms.Resample(48000, 16000)(clean_x[i, 0, :].view(1, -1))
                pred_x_16 = torchaudio.transforms.Resample(48000, 16000)(pred_x[i, 0, :].view(1, -1))

                clean_x_16 = clean_x_16.cpu().cpu().numpy()
                pred_x_16 = pred_x_16.detach().cpu().numpy()

                psq += pesq(clean_x_16.flatten(), pred_x_16.flatten(), 16000)

            psq /= len(clean_x)
            self.test_pesq += psq
            counter += 1

        self.test_pesq /= counter
        print("Value of PESQ: {:.6f}".format(self.test_pesq))

    def save(self):
        torch.save(self.model.state_dict(), self.config['save_path'])

    def load(self):
        self.model.load_state_dict(torch.load(self.config['load_path']))