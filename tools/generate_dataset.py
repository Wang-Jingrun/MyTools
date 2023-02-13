import os
import numpy as np
import random
import soundfile as sf
from numpy.linalg import norm


def signal_by_db(speech, noise, snr):
    speech = speech.astype(np.int16)
    noise = noise.astype(np.int16)

    len_speech = speech.shape[0]
    len_noise = noise.shape[0]
    start = random.randint(0, len_noise - len_speech)
    end = start + len_speech

    add_noise = noise[start:end]

    add_noise = add_noise / (norm(add_noise)) * norm(speech) / (10.0 ** (0.05 * snr))
    mix = speech + add_noise
    return mix


def get_snr():
    return random.randint(-5, 20)


if __name__ == "__main__":
    dataset_path = r"D:/PycharmProjects/DATASETS"
    clean_wavs = np.loadtxt(os.path.join(dataset_path, 'test-clean.txt'), dtype='str').tolist()
    noise_wavs = np.loadtxt(os.path.join(dataset_path, 'NOISEX92.txt'), dtype='str').tolist()

    new_dataset_name = "LibriSpeech-Noisy/test-snr=10"
    save_file = os.path.join(dataset_path, 'LibriSpeech-Noisy-test-snr=10.txt')

    serve_path = "/data1/wjr/dataset/"


    with open(save_file, 'wt') as f:
        for clean_wav in clean_wavs:

            # snr = get_snr()
            snr = 10
            # 读取干净语音
            clean_file = os.path.join(dataset_path, clean_wav)
            clean_data, fs = sf.read(clean_file, dtype='int16')

            # 读取噪声
            noise_index = random.randint(0, len(noise_wavs) - 1)
            noise_file = os.path.join(dataset_path, noise_wavs[noise_index])
            if noise_index != 0:
                noise_data, fs = sf.read(noise_file, dtype='int16')
            else:
                # babble.wav数据似乎有问题，比其他数据小得多
                noise_data, fs = sf.read(noise_file, dtype='float32')
                noise_data = noise_data * 100000

            # 生成含噪语音路径
            noisy_file = os.path.join(dataset_path, new_dataset_name, clean_wav)
            noisy_path, file_name = os.path.split(noisy_file)
            file_name = file_name[:-4] + "-" + noise_file.split('\\')[-1][:-4] + "-SNR=" + str(snr) + ".wav"
            noisy_file = os.path.join(noisy_path, file_name)
            os.makedirs(noisy_path, exist_ok=True)

            # 加噪并保存路径
            mix = signal_by_db(clean_data, noise_data, snr)
            noisy_data = np.asarray(mix, dtype=np.int16)
            sf.write(noisy_file, noisy_data, fs)
            # f.write('%s %s\n' % (noisy_file, clean_file))
            # print('%s %s\n' % (noisy_file, clean_file))
            f.write('%s %s\n' % (noisy_file.replace("D:/PycharmProjects/DATASETS", "/data1/wjr/dataset/").replace("\\", "/").replace("//", "/"),
                                 os.path.join(serve_path, clean_wav).replace("\\", "/")))
            print('%s %s\n' % (noisy_file.replace("D:/PycharmProjects/DATASETS", "/data1/wjr/dataset/").replace("\\", "/").replace("//", "/"),
                                 os.path.join(serve_path, clean_wav).replace("\\", "/")))



