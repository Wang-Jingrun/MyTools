import os
import numpy as np
import pyroomacoustics, soundfile, random


def generate_rir(room_dim, source, microphone, file_name='Room'):
    """
        room_dim: 房间尺寸
        source: 声源位置
        microphone: 麦克风位置
    """
    # 所需的混响时间和房间的尺寸
    rt60_tgt = 0.08
    # room_dim = [2, 2, 2]
    # 返回 墙壁吸收的能量 和 允许的反射次数
    e_absorption, max_order = pyroomacoustics.inverse_sabine(rt60_tgt, room_dim)

    room = pyroomacoustics.ShoeBox(room_dim, fs=16000, materials=pyroomacoustics.Material(e_absorption),
                                   max_order=max_order)
    # 在房间放置麦克风和声源
    room.add_source(source)
    room.add_microphone(microphone)

    # 创建房间冲击响应（rir）
    room.compute_rir()
    rir = room.rir[0][0]
    rir = rir[np.argmax(rir): ]

    soundfile.write(file_name + '.wav', rir, 16000)


if __name__ == '__main__':
    rooms = [
        [7, 4, 3.2], [8, 5, 3.3],
        [5.1, 3.3, 3.2], [4.8, 3.6, 3.1], [4.8, 3.3, 2.9], [4.5, 3.6, 3.1], [4.5, 3.3, 3], [3.5, 3, 2.8],
        [2.5, 2, 2.8], [2.35, 2.35, 3]
    ]
    # room_l = float(format(random.uniform(1, 20), '.2f'))
    # room_w = float(format(random.uniform(1, 20), '.2f'))
    # room_h = float(format(random.uniform(2, 5), '.2f'))
    # room_dim = [room_l, room_w, room_h]

    for index, room_dim in enumerate(rooms):
        path = os.path.join("RIR_files", f"Room-{index+1}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "room_info.txt"), 'wt', encoding='utf-8') as f:
            for i in range(100):
                source_l = float(format(random.uniform(0, room_dim[0]-0.01), '.2f'))
                source_w = float(format(random.uniform(0, room_dim[1]-0.01), '.2f'))
                source_h = float(format(random.uniform(0, room_dim[2]-0.01), '.2f'))
                source = [source_l, source_w, source_h]

                # 确保麦克风距离离声源不超过5m
                while True:
                    mic_l = float(format(random.uniform(0, room_dim[0]-0.01), '.2f'))
                    mic_w = float(format(random.uniform(0, room_dim[1]-0.01), '.2f'))
                    mic_h = float(format(random.uniform(0, room_dim[2]-0.01), '.2f'))
                    mic = [mic_l, mic_w, mic_h]

                    distance2 = (source_l - mic_l) ** 2 + (source_w - mic_w) ** 2 + (source_h - mic_h) ** 2
                    if distance2 <= 25:
                        break

                print(room_dim, source, mic)
                print()

                generate_rir(room_dim, source, mic, os.path.join(path, f"Room{index+1}-{i+1}"))
                f.write(f"Room{index+1}-samples-{i+1}: {room_dim} {source} {mic}\n")
