"""如果ffmpeg报错不存在的话，将ffmpeg.exe文件复制到当前代码的目录下即可"""
import os


def flac_to_wav(filepath, savedir):
    # 需要其他格式在这里修改即可
    filename = filepath.replace('.flac', '.wav')
    savefilename = filename.split('\\')
    save_dir = savedir + '\\' + savefilename[-1]
    # os.remove(save_dir)
    print(save_dir)
    cmd = 'ffmpeg.exe -i ' + filepath + ' ' + save_dir
    os.system(cmd)


if __name__ == '__main__':
    # 单个文件
    # audio_path = r"带转换的音频文件路径"
    # savedir = r"新保存路径"
    # flac_to_wav(audio_path, savedir)

    # 批量处理
    path = r'D:\PycharmProjects\DATASETS\LibriSpeech'
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = root + "\\" + name
            if filepath.split('.')[-1] == "flac":
                # 保存在原flac文件的位置
                os.remove(filepath)
                print(filepath)
                # flac_to_wav(filepath, root)
