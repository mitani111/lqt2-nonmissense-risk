import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from sksurv.metrics import concordance_index_censored as c_index

from config import (BRANCH_DIMS, LATENT_DIM, N_DESCRIPTORS, TRUNK_DIMS,
                    data_root, out_root, reset_dir, resolve_device, set_seed,
                    tag)
from models import (AminoDataset, CalcPartialLikelihood, DecoderCategorical,
                    Encoder, Encoder_last, Pyz, make_bool,
                    standard_normal_prior)


BATCH_SIZE = 24
LR_ENCODER = 2e-4          # branch encoders, trunk encoder and decoder
LR_SURVIVAL_HEAD = 4e-3    # DeepSurv head
N_EPOCHS = 150
SAVE_EVERY = 1

SPLITS = {
    "cv1": ("train/val1/train", "train/val1/test"),
    "cv2": ("train/val2/train", "train/val2/test"),
    "cv3": ("train/val3/train", "train/val3/test"),
    "holdout": ("test/train", "test/test"),
    "full": ("latent", None),
}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune the VAE with a DeepSurv head")
    p.add_argument("--split", choices=sorted(SPLITS), default="holdout")
    p.add_argument("--alphaZ", type=float, default=1.0, help="coefficient of the KL term")
    p.add_argument("--betaZ", type=float, default=1.0, help="coefficient of the survival term")
    p.add_argument("--pretrain-epoch", type=int, default=200,
                   help="pre-training checkpoint to start from")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--tensor-dir", default=None)
    p.add_argument("--meta-csv", default=None)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def build_loader(tensor_dir, subdir, meta_csv, transform, shuffle, drop_last):
    directory = os.path.join(tensor_dir, subdir)
    dataset = AminoDataset(file_list=os.listdir(directory),
                           dir_=directory,
                           df_path=meta_csv,
                           label_dir=os.path.join(tensor_dir, "labels"),
                           transform=transform,
                           with_clinical=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=drop_last)


def main():
    args = parse_args()

    alpha_z, beta_z = args.alphaZ, args.betaZ
    droot = data_root(args.data_root)
    oroot = out_root(args.out_root)
    tensor_dir = args.tensor_dir or os.path.join(droot, "tensors")
    meta_csv = args.meta_csv or os.path.join(droot, "in_data", "metadata.csv")

    load_path = os.path.join(oroot, f"pretrain_z{tag(alpha_z)}",
                             f"epoch_{args.pretrain_epoch}")
    save_path = os.path.join(oroot,
                             f"finetune_z{tag(alpha_z)}_{tag(beta_z)}",
                             args.split)

    device = resolve_device(args.device)
    set_seed(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_subdir, val_subdir = SPLITS[args.split]
    data_loader = build_loader(tensor_dir, train_subdir, meta_csv, transform,
                               shuffle=True, drop_last=True)
    val_loader = None
    if val_subdir is not None:
        val_loader = build_loader(tensor_dir, val_subdir, meta_csv, transform,
                                  shuffle=False, drop_last=False)

    branches = []
    for k in range(N_DESCRIPTORS):
        m = Encoder(BRANCH_DIMS[0], BRANCH_DIMS[1])
        m.load_state_dict(torch.load(os.path.join(load_path, f"encoder_1_{k}.pth"),
                                     map_location=device, weights_only=True))
        branches.append(m.to(device))

    encoder_2 = Encoder(TRUNK_DIMS[0], TRUNK_DIMS[1])
    encoder_2.load_state_dict(torch.load(os.path.join(load_path, "encoder_2.pth"),
                                         map_location=device, weights_only=True))
    encoder_2.to(device)

    encoder_3 = Encoder(TRUNK_DIMS[1], TRUNK_DIMS[2])
    encoder_3.load_state_dict(torch.load(os.path.join(load_path, "encoder_3.pth"),
                                         map_location=device, weights_only=True))
    encoder_3.to(device)

    encoder_4 = Encoder_last(TRUNK_DIMS[2], TRUNK_DIMS[3], TRUNK_DIMS[4])
    encoder_4.load_state_dict(torch.load(os.path.join(load_path, "encoder_4.pth"),
                                         map_location=device, weights_only=True))
    encoder_4.to(device)

    decoder = DecoderCategorical(input_dim=LATENT_DIM)
    decoder.load_state_dict(torch.load(os.path.join(load_path, "decoder.pth"),
                                       map_location=device, weights_only=True))
    decoder.to(device)

    pyz = Pyz(LATENT_DIM).to(device)

    modules = branches + [encoder_2, encoder_3, encoder_4, decoder, pyz]
    optimizers = ([optim.Adam(m.parameters(), lr=LR_ENCODER) for m in branches]
                  + [optim.Adam(encoder_2.parameters(), lr=LR_ENCODER),
                     optim.Adam(encoder_3.parameters(), lr=LR_ENCODER),
                     optim.Adam(encoder_4.parameters(), lr=LR_ENCODER),
                     optim.Adam(decoder.parameters(), lr=LR_ENCODER),
                     optim.Adam(pyz.parameters(), lr=LR_SURVIVAL_HEAD)])

    prior = standard_normal_prior(LATENT_DIM, device)
    recon_loss = nn.CrossEntropyLoss()

    def forward_to_risk(arrays, male, female, sample_latent=True):
        x = torch.cat([branches[k](arrays[:, :, :, k])
                       for k in range(N_DESCRIPTORS)], dim=2)
        x = encoder_2(x)
        x = encoder_3(x)
        x_mu, x_std = encoder_4(x)

        avoid_inf = torch.ones_like(x_std) * 1.0e-15
        q_z = dist.Normal(x_mu, x_std + avoid_inf, validate_args=False)
        z = q_z.rsample() if sample_latent else x_mu

        x_hat = decoder(z)
        covariates = torch.cat([male, female], dim=1)
        z_target = torch.cat([z, covariates.unsqueeze(dim=1).float()], dim=2)
        return pyz(z_target).squeeze(), x_hat, q_z

    def evaluate_val_cindex(loader):
        for m in modules:
            m.eval()
        events, times, risks = [], [], []
        with torch.no_grad():
            for arrays, _, (event, age), (male, female) in loader:
                arrays = arrays.float().to(device)
                risk, _, _ = forward_to_risk(arrays,
                                             male.float().to(device),
                                             female.float().to(device),
                                             sample_latent=False)
                events.append(event.detach().cpu().numpy().reshape(-1))
                times.append(age.detach().cpu().numpy().reshape(-1))
                risks.append(risk.detach().cpu().numpy().reshape(-1))
        events = np.concatenate(events)
        times = np.concatenate(times)
        risks = np.concatenate(risks)
        return c_index(events.astype(bool), times, risks)[0]

    def save_checkpoint(epoch):
        d = reset_dir(os.path.join(save_path, f"epoch_{epoch}"))
        for k, m in enumerate(branches):
            torch.save(m.state_dict(), os.path.join(d, f"encoder_1_{k}.pth"))
        torch.save(encoder_2.state_dict(), os.path.join(d, "encoder_2.pth"))
        torch.save(encoder_3.state_dict(), os.path.join(d, "encoder_3.pth"))
        torch.save(encoder_4.state_dict(), os.path.join(d, "encoder_4.pth"))
        torch.save(pyz.state_dict(), os.path.join(d, "pyz.pth"))

    # ---------------- training ------------------------------------------
    history = {f"{k}_{m}": [] for k in ("loss", "KLz", "CExz", "CEyz")
               for m in ("full", "epoch")}
    val_cindex_epoch = []

    for epoch in range(N_EPOCHS + 1):

        batch_loss, batch_KLz, batch_CExz, batch_CEyz = [], [], [], []

        for arrays, label, (event, age), (male, female) in data_loader:
            for m in modules:
                m.train()

            # a batch without any observed event carries no partial likelihood
            if event.sum().item() == 0:
                continue

            arrays = arrays.float().to(device)
            for opt in optimizers:
                opt.zero_grad()

            risk, x_hat, q_z = forward_to_risk(arrays,
                                               male.float().to(device),
                                               female.float().to(device),
                                               sample_latent=True)

            CEyz = CalcPartialLikelihood(risk, make_bool(event), age.squeeze())
            CExz = recon_loss(x_hat, label.to(device))
            KLz = dist.kl_divergence(q_z, prior).sum(2).mean()
            loss = KLz * alpha_z + CExz + CEyz * beta_z

            loss.backward()
            for opt in optimizers:
                opt.step()

            for key, value in (("loss", loss), ("KLz", KLz),
                               ("CExz", CExz), ("CEyz", CEyz)):
                history[f"{key}_full"].append(value.item())
            batch_loss.append(loss.item())
            batch_KLz.append(KLz.item())
            batch_CExz.append(CExz.item())
            batch_CEyz.append(CEyz.item())

        # per-epoch values are means over batches, not the last batch
        history["loss_epoch"].append(np.mean(batch_loss))
        history["KLz_epoch"].append(np.mean(batch_KLz))
        history["CExz_epoch"].append(np.mean(batch_CExz))
        history["CEyz_epoch"].append(np.mean(batch_CEyz))

        if val_loader is not None:
            val_cindex_epoch.append(evaluate_val_cindex(val_loader))
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch)

    # ---------------- outputs -------------------------------------------
    loss_path = reset_dir(os.path.join(save_path, "loss"))
    for name, values in history.items():
        np.save(os.path.join(loss_path, name), np.array(values))
    if val_loader is not None:
        np.save(os.path.join(loss_path, "val_cindex_epoch"),
                np.array(val_cindex_epoch))


if __name__ == "__main__":
    main()
