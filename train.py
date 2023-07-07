import sys

import datasets
from audiocraft.models.encodec import EncodecModel
import torch
from torch.utils.data import DataLoader
from audiocraft.data.audio_dataset import AudioDataset
import torchaudio as ta
from torch.utils.tensorboard import SummaryWriter

sample_rate = 24000
dataset = AudioDataset.from_path(root="./data/LibriSpeech",
                                 segment_duration=1,
                                 sample_rate=sample_rate,
                                 channels=2,
                                 num_samples=3000,
                                 pad=True)
trained_models_dir = "./trained_models"
writer = SummaryWriter("C:/Users/Ben/Desktop/Neuro Project/tensorboard")
# C:\Users\Ben\Desktop\Neuro Project\tensorboard

""" Hyperparams """
batch_size = 64
epochs = 20
lr = 0.0003
wd = 0

l1_loss = torch.nn.L1Loss(reduction="mean")
l2_loss = torch.nn.MSELoss()
def reconstruction_freq_loss(pred_audio, target_audio):
    window_sizes = [5, 6, 7, 8, 9, 10, 11]
    total_loss = 0
    for i in window_sizes:
        mel_spectrogram = ta.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                       n_fft=126,
                                                       normalized=True,
                                                       win_length=pow(2, i),
                                                       hop_length=pow(2, i)/4)
        pred_mel = mel_spectrogram(pred_audio)
        target_mel = mel_spectrogram(target_audio)
        total_loss += l1_loss(pred_mel[], target_mel[]) + l2_loss(pred_mel[], target_mel[])
    return total_loss * (1/len(window_sizes)*len(pred_audio))



def train_encodec(encodec: EncodecModel, device):
    encodec.train()
    encodec.requires_grad_(True)
    optimizer = torch.optim.Adam(encodec.parameters(), lr=lr, weight_decay=wd)
    # descr_optimizer = torch.optim.Adam(encodec., lr=lr, weight_decay=wd)
    # ds = Subset(dataset, torch.arange(len(dataset) - 1))
    train_set, val_set = datasets.train_test(dataset, 0.96)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # TODO - dont apply transformations to validation set
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """ Train """
    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        for i, inputs in enumerate(train_loader):
            ta.save("sample.wav", inputs[0], sample_rate)
            print("\t" + str(int((i+1) / len(train_loader) * 100)) + "%")
            loss = encodec_train_step(encodec, inputs, optimizer, device)
            writer.add_scalar("encodec loss", loss, i + epoch * len(train_loader))

        """ Validate """
        running_loss = 0
        for i, inputs in enumerate(val_loader):
            torch.cuda.empty_cache()
            inputs = inputs.to(device)
            q_res = encodec(inputs)
            running_loss += l1_loss(q_res.x, inputs).item()
        print("\tvloss: " + str(round(running_loss/len(val_loader), 4)))

        print("Saving Encodec")
        torch.save(encodec, trained_models_dir + "/encodec" + str(epoch) + ".pt")


def train_language_model(lm: torch.nn.Module):
    lm.train()


def lm_train_step():
    pass


def encodec_train_step(encodec: EncodecModel, inputs, optimizer: torch.optim.Optimizer, device, descr_optimizer: torch.optim.Optimizer = None):
    torch.cuda.empty_cache()
    inputs = inputs.to(device)
    optimizer.zero_grad()

    q_res = encodec(inputs)
    loss = l1_loss(q_res.x, inputs)
    loss.backward()

    print("\tloss: " + str(round(loss.item(), 4)))
    optimizer.step()
    return loss.item()


def add_tb_graph(train_loader, device, model):
    model.eval()
    example = None
    for i, inputs in enumerate(train_loader):
        example = inputs.to(device)
        break

    writer.add_graph(model, example)
    print("Tensorboard Graph Created")
    sys.exit(0)
