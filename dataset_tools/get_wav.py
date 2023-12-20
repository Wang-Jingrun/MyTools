import os
import numpy as np


def get_wav(base_path):
    with open(f"{base_path.split('/')[-1]}.txt", 'wt', encoding='utf-8') as f:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".wav"):
                    file_name = os.path.join(root, file)
                    f.write("%s\n"%file_name)
                    print(file_name)


if __name__ == '__main__':
    get_wav("TRAIN")
    get_wav("TEST")
