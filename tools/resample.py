import os
import librosa
import soundfile

if __name__ == '__main__':
    fs = 16000

    origin_path = './SCAFE/'
    files = os.listdir(origin_path)


    resample_path = f'./SCAFE-rs/'
    if not os.path.exists(resample_path):  # 创建保存的目录
        os.mkdir(resample_path)

    for file in files:
        if file[-4:] == '.wav':
            print(file)
            sample, _ = librosa.load(origin_path + file, sr=fs)
            soundfile.write(resample_path + file[:-4] + '.wav', sample, fs)
