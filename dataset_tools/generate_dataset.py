import os
import numpy as np
import random, soundfile
from numpy.linalg import norm


def load_wav(wav_file):
    wav, sr = soundfile.read(wav_file)
    wav = wav.astype('float32')
    # 能量归一化
    wav = wav / ((np.sqrt(np.sum(wav ** 2)) / ((wav.size) + 1e-7)) + 1e-7)
    return wav


class CreateDataset:
    def __init__(self, dataset_path, clean_files, noise_files, rir_files):
        super(CreateDataset, self).__init__()
        self.dataset_path = dataset_path
        self.clean_files = np.loadtxt(os.path.join(dataset_path, clean_files), dtype='str').tolist()
        self.noise_files = np.loadtxt(os.path.join(dataset_path, noise_files), dtype='str').tolist()
        self.rir_files = np.loadtxt(os.path.join(dataset_path, rir_files), dtype='str').tolist()
        self.num = len(self.clean_files)

    def generate_wavs(self, dataset_newname='Evaluation/LibriSpeech-test-snr=0', snr=0):
        with open(dataset_newname.split('/')[-1] + '.txt', 'wt') as f:
            for i in range(self.num):
                # 读取干净语音
                clean_file = os.path.join(self.dataset_path, self.clean_files[i])
                clean_wav = load_wav(clean_file)

                rir_wav = self.add_rir(clean_wav)
                noisy_wav, noise_name = self.add_noise(rir_wav, snr)

                # 生成含噪语音路径
                noisy_file = os.path.join(self.dataset_path, dataset_newname, self.clean_files[i])
                noisy_path, file_name = os.path.split(noisy_file)
                file_name = file_name[:-4] + "-" + noise_name + "-SNR=" + str(snr) + ".wav"
                noisy_file = os.path.join(noisy_path, file_name)
                os.makedirs(noisy_path, exist_ok=True)

                soundfile.write(noisy_file, noisy_wav.astype('int16'), 16000)

                f.write('%s %s\n' % (noisy_file, clean_file))
                print('%s %s\n' % (noisy_file, clean_file))

    def add_rir(self, wav):
        rir_file = random.choice(self.rir_files)
        rir, _ = soundfile.read(os.path.join(self.dataset_path, rir_file))

        out_wav = np.convolve(wav, rir)
        return out_wav[:wav.shape[0]]

    def add_noise(self, wav, snr=0):
        noise_file = random.choice(self.noise_files)
        noise = load_wav(os.path.join(self.dataset_path, noise_file))

        # 对噪声进行裁剪
        len_speech = wav.shape[0]
        len_noise = noise.shape[0]
        # 噪声文件前面一段时间可能没有声音
        start = random.randint(1000, len_noise - len_speech)
        noise = noise[start: start + len_speech]

        add_nosie = noise / norm(noise) * norm(wav) / (10.0 ** (0.05 * snr))
        return wav + add_nosie, noise_file.split('/')[-1][:-4]


if __name__ == "__main__":
    dataset_path = r"D:/PycharmProjects/DATASETS"
    clean_files = "test-clean.txt"
    noise_files = "test-noise.txt"
    rir_files = "RIR_files_test.txt"

    createdataset = CreateDataset(dataset_path, clean_files, noise_files, rir_files)
    createdataset.generate_wavs(dataset_newname='Evaluation/LibriSpeech-test-snr=0', snr=0)
    createdataset.generate_wavs(dataset_newname='Evaluation/LibriSpeech-test-snr=5', snr=5)
    createdataset.generate_wavs(dataset_newname='Evaluation/LibriSpeech-test-snr=10', snr=10)