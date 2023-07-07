import torchaudio as ta
from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_init_encodec, load_lm_model
import os
import torch
import train
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_encoder():
    encodec = torch.load("./trained_models/encodec6.pt")
    encodec.eval()
    audio = ta.load("./data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac")
    ta.save("sample.wav", audio[0], train.sample_rate)
    audio = audio[0]#[None, :, :]
    audio = torch.stack([audio, audio], 1)
    audio = audio.to(device)
    q_res = encodec(audio)
    res = q_res.x.to("cpu")
    res = res.detach()
    res = res + 0.5
    ta.save("encodec_output.wav", res[0], train.sample_rate)


def train_models():
    cache_dir = os.environ.get('MUSICGEN_ROOT', None)
    name = "small"

    datasets.clear_folder("./tensorboard")

    encodec = load_init_encodec(name, device=device, sample_rate=train.sample_rate, cache_dir=cache_dir)
    train.train_encodec(encodec, device)

    lm = load_lm_model(name, device=device, cache_dir=cache_dir)

    model = MusicGen(name=name, compression_model=encodec, lm=lm)

    # output = model.generate(
    #     descriptions=[
    #         '80s pop track with bassy drums and synth',
    #     ],
    #     progress=True
    # )
    #
    # output = output.to("cpu")
    # ta.save("my_output.wav", output[0], train.sample_rate)


def main():
    # test_encoder()
    train_models()


if __name__ == "__main__":
    main()
