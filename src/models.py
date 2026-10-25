import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import Dataset

from config import MSA_LENGTH, N_AA_CLASSES

def array_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".npy"))

class AminoDataset(Dataset):
    def __init__(self, file_list, dir_, df_path, label_dir,
                 transform=None, with_clinical=False):
        self.file_list = list(file_list)
        self.dir = dir_
        self.labeldir = label_dir
        self.transform = transform
        self.with_clinical = with_clinical
        self.df = pd.read_csv(df_path)
        if with_clinical:
            self._meta = self.df.set_index("ID")[
                ["Event", "Age", "Male", "Female"]
            ]

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def _record_id(filename):
        # "array_<ID>.npy" -> <ID>
        return int(filename[6:-4])

    def __getitem__(self, idx):
        name = self.file_list[idx]
        array = np.load(f"{self.dir}/{name}")
        label = np.load(f"{self.labeldir}/{name}")

        if self.transform is not None:
            array = self.transform(array)
        # CrossEntropyLoss requires integer class targets.
        label = torch.tensor(label).long()

        if not self.with_clinical:
            return array, label

        row = self._meta.loc[self._record_id(name)]
        event = np.asarray([row["Event"]])
        age = np.asarray([row["Age"]])
        male = np.asarray([row["Male"]])
        female = np.asarray([row["Female"]])
        return array, label, (event, age), (male, female)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.fc(x))


class Encoder_last(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super().__init__()
        self.fc_mu1 = nn.Linear(input_dim, mid_dim)
        self.fc_mu2 = nn.Linear(mid_dim, output_dim)
        self.fc_std1 = nn.Linear(input_dim, mid_dim)
        self.fc_std2 = nn.Linear(mid_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        mu = self.softplus(self.fc_mu1(x))
        mu = self.fc_mu2(mu)
        std = self.softplus(self.fc_std1(x))
        std = self.softplus(self.fc_std2(std))
        return mu, std


class DecoderCategorical(nn.Module):
    def __init__(self, input_dim=100, hidden_dims=(256, 512, 1024), rank=128, seq_length=1380, num_aa_types=21):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.down = nn.Linear(h3, rank)
        self.up = nn.Linear(rank, seq_length * num_aa_types)
        self.seq_length, self.num_aa_types = seq_length, num_aa_types
        self.act = nn.Softplus()
 
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.up(self.down(x))
        return x.view(-1, self.num_aa_types, self.seq_length)

class Pyz(nn.Module):
    def __init__(self, input_dim, n_covariates=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + n_covariates, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def make_bool(event):
    return [bool(event[i] != 0) for i in range(len(event))]


def CalcPartialLikelihood(hazard, E, T):
    log_likelihood = torch.tensor(0.0)
    index = torch.tensor(np.arange(len(T)))
    event_time = T[E]
    event_index = index[E]
    for i in range(len(event_time)):
        the_T = event_time[i]
        the_hazard = hazard[event_index[i]]
        base = hazard[T >= the_T]
        base = torch.log(torch.exp(base).sum())
        log_likelihood = log_likelihood + the_hazard - base
    return -log_likelihood / len(event_time)


def standard_normal_prior(dim, device):
    return dist.Normal(torch.zeros([dim]).to(device),
                       torch.ones([dim]).to(device))
