import random


def shuffle_lines_in_file(filename):
    # 读取文本文件并将每行存储在列表中
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 随机打乱列表
    random.shuffle(lines)

    # 将打乱后的列表写回文本文件
    with open(filename, 'w') as file:
        file.writelines(lines)


# 调用示例函数，传入文本文件路径
shuffle_lines_in_file('TIMIT_train_s.txt')
