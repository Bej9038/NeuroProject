import torchaudio
import datasets
from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_init_encodec, load_lm_model
from audiocraft.models.encodec import EncodecModel
import os
import torch
from torch.utils.data import Subset, DataLoader

batch_size = 40
epochs = 20
lr = 0.01
wd = 0
dataset = datasets.KshmrVol3()
loss_fn = torch.nn.CrossEntropyLoss()


def train_encodec(encodec: EncodecModel):
    encodec.train()
    optimizer = torch.optim.Adam(encodec.parameters(), lr=lr, weight_decay=wd)
    # descr_optimizer = torch.optim.Adam(encodec, lr=lr, weight_decay=wd)
    ds = Subset(dataset, torch.arange(len(dataset) - 1))
    train_set, val_set = datasets.train_test(ds, 0.8)
    # collate_fn = pad_to_longest_fn
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    """ Train """
    for epoch in range(epochs):
        for i, (audio_batch, descriptions) in enumerate(train_loader):
            encodec_train_step(encodec, optimizer, descr_optimizer, audio_batch)
        # evaluate


def train_language_model(lm: torch.nn.Module):
    lm.train()


def lm_train_step():


def encodec_train_step(encodec: EncodecModel, optimizer: torch.optim.Optimizer, descr_optimizer: torch.optim.Optimizer,
                       audio_batch):
    optimizer.zero_grad()
    descr_optimizer.zero_grad()

    output = encodec(audio_batch)
    loss = 0
    loss.backward()

    optimizer.step()
    descr_optimizer.step()
