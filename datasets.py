import os
import pandas as pd
import shutil
import re
import torchaudio
from torch.utils.data import Dataset, random_split


class KshmrVol3(Dataset):
    def __init__(self):
        self.directory = "./data/kshmr_data/audio_files"
        self.text_labels = pd.read_csv("./data/kshmr_data/descriptions", delimiter=' : ', header=None)

    def __len__(self):
        return len(self.text_labels)

    def __getitem__(self, index):
        path = self.directory + "/" + self.text_labels.iloc[index, 0]
        audio = torchaudio.load(path)
        label = self.text_labels.iloc[index, 1]
        return audio, label


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def train_test(dataset, split):
    train_size = int(split * len(dataset))
    test_size = (len(dataset) - train_size)
    return random_split(dataset, [int(train_size), int(test_size)])


def main():
    dir = "./data/kshmr_data/audio_files"
    # dest = "./data/kshmr_data/audio_files"
    textfile = "./data/kshmr_data/descriptions"
    with open(textfile, 'w') as f:
        f.truncate()
    with open(textfile, 'a') as f:
        for root, _, files in os.walk(dir):
            for file in files:
                f.write(file + " : " + re.sub(r'\d+', '', file[6:-4])
                        .replace("___", " ")
                        .replace("__", " ")
                        .replace("_", " ") + "\n")


if __name__ == "__main__":
    main()
