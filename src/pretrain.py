import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (BRANCH_DIMS, LATENT_DIM, N_DESCRIPTORS, TRUNK_DIMS,
                    data_root, out_root, reset_dir, resolve_device, set_seed,
                    tag)
from models import (AminoDataset, DecoderCategorical, Encoder, Encoder_last,
                    standard_normal_prior)


BATCH_SIZE = 34
LEARNING_RATE = 1e-3
N_CHECKPOINTS = 20
SAVE_EVERY = 20


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train the VAE on variant sequences")
    p.add_argument("--alphaZ", type=float, default=1.0,
                   help="coefficient of the KL term")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--tensor-dir", default=None,
                   help="directory holding all/ and labels/ (default <data-root>/tensors)")
    p.add_argument("--meta-csv", default=None,
                   help="metadata table (default <data-root>/in_data/metadata.csv)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    start_cpu_time = time.process_time()
    start_time = time.perf_counter()

    alpha_z = args.alphaZ
    droot = data_root(args.data_root)
    oroot = out_root(args.out_root)
    tensor_dir = args.tensor_dir or os.path.join(droot, "tensors")
    meta_csv = args.meta_csv or os.path.join(droot, "in_data", "metadata.csv")
    save_path = os.path.join(oroot, f"pretrain_z{tag(alpha_z)}")

    device = resolve_device(args.device)
    set_seed(device)

    # ---------------- dataset -------------------------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    all_dir = os.path.join(tensor_dir, "all")
    dataset = AminoDataset(file_list=os.listdir(all_dir),
                           dir_=all_dir,
                           df_path=meta_csv,
                           label_dir=os.path.join(tensor_dir, "labels"),
                           transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=True, drop_last=True)

    # ---------------- model & optimizers --------------------------------
    branches = [Encoder(BRANCH_DIMS[0], BRANCH_DIMS[1]).to(device)
                for _ in range(N_DESCRIPTORS)]
    encoder_2 = Encoder(TRUNK_DIMS[0], TRUNK_DIMS[1]).to(device)
    encoder_3 = Encoder(TRUNK_DIMS[1], TRUNK_DIMS[2]).to(device)
    encoder_4 = Encoder_last(TRUNK_DIMS[2], TRUNK_DIMS[3], TRUNK_DIMS[4]).to(device)
    decoder = DecoderCategorical(input_dim=LATENT_DIM).to(device)

    opt_branches = [optim.Adam(m.parameters(), lr=LEARNING_RATE) for m in branches]
    opt_en_2 = optim.Adam(encoder_2.parameters(), lr=LEARNING_RATE)
    opt_en_3 = optim.Adam(encoder_3.parameters(), lr=LEARNING_RATE)
    opt_en_4 = optim.Adam(encoder_4.parameters(), lr=LEARNING_RATE)
    opt_de = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    def save_checkpoint(epoch):
        d = reset_dir(os.path.join(save_path, f"epoch_{epoch}"))
        for k, m in enumerate(branches):
            torch.save(m.state_dict(), os.path.join(d, f"encoder_1_{k}.pth"))
        torch.save(encoder_2.state_dict(), os.path.join(d, "encoder_2.pth"))
        torch.save(encoder_3.state_dict(), os.path.join(d, "encoder_3.pth"))
        torch.save(encoder_4.state_dict(), os.path.join(d, "encoder_4.pth"))
        torch.save(decoder.state_dict(), os.path.join(d, "decoder.pth"))

    # ---------------- training ------------------------------------------
    n_epochs = SAVE_EVERY * N_CHECKPOINTS + 1
    prior = standard_normal_prior(LATENT_DIM, device)
    recon_loss = nn.CrossEntropyLoss()

    loss_epoch, KLz_epoch, CExz_epoch = [], [], []
    loss_all, KLz_all, CExz_all = [], [], []

    for epoch in range(n_epochs):

        for arrays, labels in data_loader:
            arrays = arrays.float().to(device)

            for opt in opt_branches:
                opt.zero_grad()
            opt_en_2.zero_grad()
            opt_en_3.zero_grad()
            opt_en_4.zero_grad()
            opt_de.zero_grad()

            x = torch.cat([branches[k](arrays[:, :, :, k])
                           for k in range(N_DESCRIPTORS)], dim=2)
            x = encoder_2(x)
            x = encoder_3(x)
            x_mu, x_std = encoder_4(x)

            avoid_inf = torch.ones_like(x_std) * 1.0e-9
            q_z = dist.Normal(x_mu, x_std + avoid_inf, validate_args=False)
            z = q_z.rsample()
            x_hat = decoder(z)

            CExz = recon_loss(x_hat, labels.to(device))
            KLz = dist.kl_divergence(q_z, prior).sum(2).mean()
            loss = alpha_z * KLz + CExz
            loss.backward()

            for opt in opt_branches:
                opt.step()
            opt_en_2.step()
            opt_en_3.step()
            opt_en_4.step()
            opt_de.step()

            loss_all.append(loss.item())
            KLz_all.append(KLz.item() * alpha_z)
            CExz_all.append(CExz.item())

        loss_epoch.append(loss.item())
        KLz_epoch.append(KLz.item())
        CExz_epoch.append(CExz.item())

    loss_path = reset_dir(os.path.join(save_path, "loss"))
    for name, values in [("loss_epoch", loss_epoch), ("KLz_epoch", KLz_epoch),
                         ("CExz_epoch", CExz_epoch), ("loss_all", loss_all),
                         ("KLz_all", KLz_all), ("CExz_all", CExz_all)]:
        np.save(os.path.join(loss_path, name), np.array(values))
    if epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch)

if __name__ == "__main__":
    main()
