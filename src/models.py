import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import Dataset

from config import MSA_LENGTH, N_AA_CLASSES

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
    def __init__(self, input_dim, seq_length=MSA_LENGTH,
                 num_aa_types=N_AA_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 8000)
        self.fc3 = nn.Linear(8000, 20000)
        self.fc4 = nn.Linear(20000, seq_length * num_aa_types)

        self.seq_length = seq_length
        self.num_aa_types = num_aa_types
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        x = self.softplus(self.fc4(x))
        x = x.view(-1, self.num_aa_types, self.seq_length)
        return torch.softmax(x, dim=-1)

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
