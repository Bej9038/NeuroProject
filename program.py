import torchaudio
from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_init_encodec, load_lm_model
import os
import torch
import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cache_dir = os.environ.get('MUSICGEN_ROOT', None)
    name = "small"

    encodec = load_init_encodec(name, device=device, sample_rate=train.sample_rate, cache_dir=cache_dir)
    train.train_encodec(encodec, device)

    lm = load_lm_model(name, device=device, cache_dir=cache_dir)

    model = MusicGen(name=name, compression_model=encodec, lm=lm)

    output = model.generate(
        descriptions=[
            '80s pop track with bassy drums and synth',
        ],
        progress=True
    )

    output = output.to("cpu")
    torchaudio.save("my_output.wav", output[0], train.sample_rate)


if __name__ == "__main__":
    main()
