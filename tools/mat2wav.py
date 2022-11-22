import os
from scipy.io import loadmat
from scipy.io import wavfile

if __name__ == '__main__':
    mat_path = './mat/'
    files = os.listdir(mat_path)

    fs = 16000
    wav_path = f'./wav-fs{fs}/'
    if not os.path.exists(wav_path):  # 创建保存的目录
        os.mkdir(wav_path)

    for file in files:
        if file[-4:] == '.mat':
            print(file)
            mat = loadmat(mat_path + file)
            data = mat[file[:-4]]
            wavfile.write(wav_path + file[:-4] + '.wav', fs, data)
