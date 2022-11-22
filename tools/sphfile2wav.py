from sphfile import SPHFile
import glob
import os

#  将TIMIT语料库转换为wav格式
#  下载好的TIMIT语料库，不能直接读取和打开因为它是sphere格式
#  要转换成wav格式才能读取和打开，下面就是转化的代码

if __name__ == "__main__":
    # path = r'SX274.WAV'
    # 多条语音转换
    path = r"D:\PycharmProjects\DATASETS\TIMIT\*\*\*\*.WAV"

    sph_files = glob.glob(path)
    print(sph_files)
    print(len(sph_files), "个语音")

    for i in sph_files:

        sph = SPHFile(i)
        #  改名字，直接适用WAV到wav，不改名字，不成功
        sph.write_wav(filename=i.replace(".WAV", "a.wav"))

        os.remove(i)  # 转换后，删除原始的语音文件

        # 重命名
        os.rename(i.replace(".WAV", "a.wav"), i.replace(".WAV", ".wav"))

    print("处理完成！")
