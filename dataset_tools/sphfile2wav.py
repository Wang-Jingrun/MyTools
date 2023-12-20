from sphfile import SPHFile
import glob
import os

#  将TIMIT语料库转换为wav格式
#  下载好的TIMIT语料库，不能直接读取和打开因为它是sphere格式
#  要转换成wav格式才能读取和打开，下面就是转化的代码

if __name__ == '__main__':
    path = "E:\PycharmProjects\DATASETS\TIMIT\*\*\*\*.WAV"
    sph_file = glob.glob(path)
    print(len(sph_file), "train utterences")
    for i in sph_file:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV", "_.wav"))
        os.remove(i)

    print("Completed!")
