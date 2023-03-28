import os
import librosa
import soundfile

if __name__ == '__main__':
    fs = 16000

    origin_path = './test_noisy_speech'
    files = os.listdir(origin_path)

    resample_path = f'./test_noisy_speech-rs'
    if not os.path.exists(resample_path):  # 创建保存的目录
        os.mkdir(resample_path)

    for file in files:
        if file[-4:] == '.wav':
            print(file)
            sample, _ = librosa.load(os.path.join(origin_path, file), sr=fs)
            soundfile.write(os.path.join(resample_path, file[:-4] + '.wav'), sample, fs)
