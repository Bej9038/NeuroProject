import torchaudio as ta
import torch
import train
import os
from audiocraft.models.loaders import load_compression_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_encoder():
    encodec = torch.load("./trained_models/encodec0.pt")
    # encodec = load_compression_model("small", device, os.environ.get('MUSICGEN_ROOT', None))
    encodec.eval()
    audio = ta.load("./data/LibriSpeech/train-clean-100/4830/25898/4830-25898-0015.flac")
    sr = audio[1]
    ta.save("./results/sample.wav", audio[0], audio[1])
    audio = audio[0]
    audio = torch.stack([audio, audio], 1)
    audio = audio.to(device)
    q_res = encodec(audio)
    res = q_res.x.to("cpu")
    res = res.detach()
    res = res + 0.5
    ta.save("./results/encodec_output.wav", res[0], sr)


def main():
    test_encoder()


if __name__ == "__main__":
    main()
