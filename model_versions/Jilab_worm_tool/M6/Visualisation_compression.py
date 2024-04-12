# Input:拍摄视频或者图片，位置CSV文件
# Output：抽帧视频（轨迹画在原视频上）
# jump_frame为视频压缩跳帧，可根据情况调整，默认50.
# 创建日期2023.9.21
# 修改时间：2023.10.17
# 修改人：安佳晖

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import colorsys
import random
import itertools
import multiprocessing

single_track = 0
def generate_colors(n):
    colors = []
    hue = 1.0
    saturation = 1.0
    value = 1.0
    step = 1.0 / n
    for _ in range(n):
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = [int(x * 255) for x in rgb]
        colors.append((r, g, b))
        hue += step
    return colors

def Vis_pic_track (root):
    root.destroy()
    # 0.09
    jump_frame=50   #跳帧数
    t_window=1000    #轨迹窗口大小
    print(type(jump_frame),type(t_window))
    print("数据类型为图片数据")
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(title='请选择一个csv文件',filetypes = (("CSV Files","*.csv"),))
    pic_path = filedialog.askdirectory(title='请选择一个图片路径文件')
    df = pd.read_csv(csv_path)
    df['Coordinates'] = df.apply(lambda row: (int(row['X']), int(row['Y'])), axis=1)
    end_time = max(df['Timestamp'].unique())
    image_files = sorted([f for f in os.listdir(pic_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    last_img_mtime = datetime.fromtimestamp(os.path.getmtime(pic_path + '/' + image_files[0]))
    colors = generate_colors(20)
    circular_iterator = itertools.cycle(colors)
    color_dict=dict(zip(list(df.ID.unique()),circular_iterator))
    image = cv2.imread(pic_path + '/' + image_files[0])
    H, W, L = image.shape
    print('生成视频')
    print(csv_path[:-4])
    def image_generator(image_files, pic_path, end_time, jump_frame):
        for j in range(0, end_time, jump_frame):
            image = cv2.imread(pic_path + '/' + image_files[j])
            current_img_mtime = datetime.fromtimestamp(os.path.getmtime(pic_path + '/' + image_files[j]))
            time_interval = current_img_mtime - last_img_mtime
            # 读取全部图片
            image = image + image
            # 正片叠底以加深
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            BI = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # 把图片转换为BGR模式
            before_df_id = df[((df['Timestamp'] > (j - (t_window / 2)))) & ((df['Timestamp'] < j))]
            # 将当前时间窗内的路径信息存到一个新数据框中
            for id in before_df_id['ID'].unique():
                before_df_plot = before_df_id[before_df_id['ID'] == id]
                cv2.polylines(image, np.int32([before_df_plot['Coordinates'].tolist()]), False, color_dict[id], 2)
                cv2.putText(image, str(before_df_plot['ID'].tolist()[0]), before_df_plot['Coordinates'].tolist()[0],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            overlap = cv2.addWeighted(BI, 0.2, image, 0.8, 0)
            cv2.putText(overlap, str(time_interval)[:-7],
                        (50, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
            yield overlap

    output_video = cv2.VideoWriter(str(csv_path[:-4]) + 'track_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))

    with tqdm(total=end_time) as pbar:
        for frame in image_generator(image_files, pic_path, end_time, jump_frame):
            output_video.write(frame)
            pbar.update(jump_frame)

    output_video.release()
    exit()

def Vis_vid_track(video_path):
    jump_frame = 50  # 跳帧数
    t_window = 1000  # 轨迹窗口大小
    csv_path = str(video_path[:-4]) + '.csv'
    df = pd.read_csv(csv_path)
    df['Coordinates'] = df.apply(lambda row: (int(row['X']), int(row['Y'])), axis=1)
    end_time = max(df['Timestamp'].unique())
    # 从时间窗一半开始，遍历到最后
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()
    H, W, L = frame.shape
    colors = generate_colors(20)
    circular_iterator = itertools.cycle(colors)
    color_dict = dict(zip(list(df.ID.unique()), circular_iterator))
    print('生成视频')
    print(csv_path[:-4])

    def image_generator(video, end_time, jump_frame):
        for j in range(0, end_time, jump_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, j)
            seconds = video.get(cv2.CAP_PROP_POS_MSEC) // 1000
            ok, frame = video.read()
            image = frame
            image = image * 2
            # 把图片转换为BGR模式
            before_df_id = df[((df['Timestamp'] > (j - (t_window / 2)))) & ((df['Timestamp'] < j))]
            for id in before_df_id['ID'].unique():
                before_df_plot = before_df_id[before_df_id['ID'] == id]
                cv2.polylines(image, np.int32([before_df_plot['Coordinates'].tolist()]), False, color_dict[id], 2)
                cv2.putText(image, str(before_df_plot['ID'].tolist()[0]), before_df_plot['Coordinates'].tolist()[0],
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            overlap = cv2.addWeighted(image, 0.2, image, 0.8, 0)
            cv2.putText(overlap, str(int(seconds / 3600)) + ':'
                        + str(int(seconds / 60 % 60)) + ':' +
                        str(int(seconds % 60)),
                        (50, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
            yield overlap

    output_video = cv2.VideoWriter(str(csv_path[:-4]) + '_track_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                   (W, H))

    with tqdm(total=end_time) as pbar:
        for frame in image_generator(video, end_time, jump_frame):
            output_video.write(frame)
            pbar.update(jump_frame)

    output_video.release()
    exit()


def Vis_multi_vid(root):
    root.destroy()
    path = filedialog.askdirectory()
    files = [f for f in os.listdir(path) if f.endswith(('.avi', '.mp4'))]
    # files = [f for f in os.listdir(path) if f.endswith(('.jpeg'))]
    print(f'共{len(files)}个视频')
    print(files)
    pool = multiprocessing.Pool(processes=len(files))
    # # 并行处理每个视频文件
    print(f'开启{len(files)}个进程')
    results = [pool.apply_async(Vis_vid_track, args=(os.path.join(path, files_path),)) for files_path in files]

    # # 等待所有进程完成
    pool.close()
    pool.join()
    exit()


def compress_vid(video_path):

    print("压缩视频数据")
    jump_frame = 200  # 跳帧数
    # root = tk.Tk()
    # root.withdraw()
    # video_path = filedialog.askopenfilename(title='请选择一个视频路径文件')
    print(video_path)
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()

    end_time = int(video.get(7))
    H, W, L = frame.shape
    video_name=video_path[:-4]

    print('生成视频')
    def image_generator(video, end_time, jump_frame):

        for j in range(0, end_time, jump_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, j)
            seconds = video.get(cv2.CAP_PROP_POS_MSEC) // 1000
            ok, frame = video.read()
            overlap = frame * 2
            cv2.putText(overlap, str(int(seconds / 3600)) + ':'
                        + str(int(seconds / 60 % 60)) + ':' +
                        str(int(seconds % 60)),
                        (50, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
            yield overlap

    output_video = cv2.VideoWriter(str(video_name) + '_zip.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                   (W, H))

    with tqdm(total=end_time) as pbar:
        for frame in image_generator(video, end_time, jump_frame):
            output_video.write(frame)
            pbar.update(jump_frame)

    output_video.release()
    # exit()


def compress_multi_vid(root):
    root.destroy()
    path = filedialog.askdirectory()
    files = [f for f in os.listdir(path) if f.endswith(('.avi', '.mp4'))]
    # files = [f for f in os.listdir(path) if f.endswith(('.jpeg'))]
    print(f'共{len(files)}个视频')
    print(files)
    pool = multiprocessing.Pool(processes=len(files))
    # # 并行处理每个视频文件
    print(f'开启{len(files)}个进程')
    results = [pool.apply_async(compress_vid, args=(os.path.join(path, files_path),)) for files_path in files]

    # # 等待所有进程完成
    pool.close()
    pool.join()
    exit()



if __name__ == "__main__":
    root = tk.Tk()
    root.title("数据类型选择")
    # jump_frame = 100
    # default_jump_frame = 100
    entry1 = tk.Entry(root)
    # 创建按钮1
    button1 = tk.Button(root, text="数据类型为图片", command=lambda: case1(root),width=30, height=5)
    button1.pack()

    # 创建按钮2
    button2 = tk.Button(root, text="数据类型为视频", command=lambda: case2(root),width=30, height=5)
    button2.pack()

    # 创建按钮3
    button3 = tk.Button(root, text="压缩视频", command=lambda: compress_vid(root), width=30, height=5)
    button3.pack()



    # 主循环
    root.mainloop()


