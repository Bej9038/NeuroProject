import torchaudio
import datasets
from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_init_encodec, load_lm_model
from audiocraft.models.encodec import EncodecModel
import os
import torch
from torch.utils.data import Subset, DataLoader
from audiocraft.data.audio_dataset import AudioDataset

batch_size = 10
epochs = 1
lr = 0.01
wd = 0
sample_rate = 44100
dataset = AudioDataset.from_path(root="./data/kshmr_data/audio_files",
                                 segment_duration=4,
                                 sample_rate=sample_rate,
                                 channels=2,
                                 pad=True)
recon_time_loss = torch.nn.L1Loss()
recon_freq_loss = torch.nn.MSELoss()


def train_encodec(encodec: EncodecModel, device):
    encodec.train()
    optimizer = torch.optim.Adam(encodec.parameters(), lr=lr, weight_decay=wd)
    # descr_optimizer = torch.optim.Adam(encodec., lr=lr, weight_decay=wd)
    # ds = Subset(dataset, torch.arange(len(dataset) - 1))
    train_set, val_set = datasets.train_test(dataset, 0.8)
    # collate_fn = pad_to_longest_fn
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # TODO - dont apply transformations to validation set
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    """ Train """
    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        for i, audio_batch in enumerate(train_loader):
            print("\t" + str((i+1) * batch_size / len(train_loader) * 100) + "%")
            encodec_train_step(encodec, audio_batch, optimizer, device)


def train_language_model(lm: torch.nn.Module):
    lm.train()


def lm_train_step():
    pass


def encodec_train_step(encodec: EncodecModel, inputs, optimizer: torch.optim.Optimizer, device, descr_optimizer: torch.optim.Optimizer = None):
    torch.cuda.empty_cache()
    inputs = inputs.to(device)
    optimizer.zero_grad()
    q_res = encodec(inputs)
    loss = recon_time_loss(q_res.x, inputs)
    loss.backward()
    print("loss: " + str(loss.item()))
    optimizer.step()
