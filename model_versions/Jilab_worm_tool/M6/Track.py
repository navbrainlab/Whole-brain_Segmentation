import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from scipy.spatial import distance_matrix
import time
import math
import sys
import datetime
from scipy.optimize import linear_sum_assignment
import csv
from operator import itemgetter
import matplotlib
import concurrent.futures
from threading import Lock
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing
from sys import exit
from matplotlib.lines import Line2D

def video_file(root, P1, P2, sum_P1, sum_P2, min_wormArea, max_wormArea, sum_jump, track_jump, show_value):
    root.destroy()
    path = filedialog.askdirectory()
    files = [f for f in os.listdir(path) if
             f.endswith(('.avi', '.mp4')) and not f.endswith('zip.avi') and not f.endswith('zip.mp4')]
    print(f'共{len(files)}个视频')
    print(files)
    pool = multiprocessing.Pool(processes=len(files))
    # # 并行处理每个视频文件
    print(f'开启{len(files)}个进程')
    results = [pool.apply_async(track_video_file_vid, args=(
    os.path.join(path, files_path), P1, P2, sum_P1, sum_P2, min_wormArea, max_wormArea, sum_jump, track_jump,
    show_value,)) for files_path in files]
    # # 等待所有进程完成
    pool.close()
    pool.join()
    exit()
def process_frame(video_path, start_frame, end_frame, skip_jump_frame, sum_P1, sum_P2, kernel, kernel_close,
                  new_sheet):
    video = cv2.VideoCapture(video_path)
    timeC = start_frame
    sum_track_local = np.zeros_like(new_sheet)
    ok = True
    while ok and timeC < end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, timeC)
        ok, frame = video.read()
        if not ok:
            break
        gray = frame[:, :, 2]
        gray = cv2.bitwise_and(gray, new_sheet)
        Canny_img = cv2.Canny(gray, sum_P1, sum_P2)
        image1 = cv2.dilate(Canny_img, kernel, iterations=1)
        im_out = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, kernel_close)
        sum_track_local += im_out
        sum_track_local[sum_track_local > 0] = 255
        timeC += skip_jump_frame
    video.release()

    return sum_track_local

def multithreaded_frame_processing(video_path, total_frames, num_threads, skip_jump_frame, sum_P1, sum_P2,
                                   kernel, kernel_close, new_sheet):
    frames_per_chunk = total_frames // num_threads
    lock = Lock()
    sum_track = np.zeros_like(new_sheet)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start_frame = i * frames_per_chunk
            end_frame = (i + 1) * frames_per_chunk
            future = executor.submit(process_frame, video_path, start_frame, end_frame, skip_jump_frame, sum_P1,
                                     sum_P2, kernel, kernel_close, new_sheet)
            futures.append(future)
        with tqdm(total=num_threads, desc="Processing Chunks", unit="chunk") as pbar:
            for future in concurrent.futures.as_completed(futures):
                sum_track_local = future.result()
                with lock:
                    sum_track += sum_track_local
                    sum_track[sum_track > 0] = 255
                pbar.update(1)

    return sum_track

def track_video_file_vid(path, P1, P2, sum_P1, sum_P2, min_wormArea, max_wormArea, sum_jump, track_jump, show_value):
    circle_info = []

    # matplotlib.use('tkagg')
    print(f'开始追踪文件{os.path.basename(path)}')

    class Target:
        def __init__(self, tid, position, frame_number):
            self.tid = tid
            self.positions = [(position, frame_number)]  # List of tuples (position, frame_number)
            self.missing_frames = 0
            self.active = True
            self.total_distance = 0
            self.track_misalignment_rate = 0
            self.track_misalignment_area = 0

        def update_position(self, position, frame_number):
            if self.active:
                # self.total_distance += np.sqrt((position[0] - self.positions[-1][0][0])**2 +
                #                                (position[1] - self.positions[-1][0][1])**2)
                self.positions.append((position, frame_number))
                # self.track_misalignment_area = len(set(list(map(itemgetter(0), self.positions))))
                # self.track_misalignment_rate=(len(set(list(map(itemgetter(0), self.positions)))))/len(self.positions)
            self.missing_frames = 0

        def increase_missing_frames(self):
            self.missing_frames += 1

        def check_death(self):
            if self.missing_frames >= max_missing_frames:
                self.active = False

        def calculate_distance(self, other_target):
            last_position_self = self.positions[-1][0]
            last_position_other = other_target.positions[-1][0]
            distance = np.sqrt((last_position_self[0] - last_position_other[0]) ** 2 +
                               (last_position_self[1] - last_position_other[1]) ** 2)
            return distance

    def on_click(event):
        global circle
        if event.button == 1:  # 鼠标左键点击事件
            if len(points) == 3:
                points.clear()
            # ax = plt.gca()
            # ax.plot(event.xdata, event.ydata, 'ro')  # 以红色圆点标记点击的点
            points.append((event.xdata, event.ydata))
            line.set_data(*zip(*points))
            plt.draw()
            if len(points) == 3:
                circle = draw_circle()
                # points.clear()
        elif event.button == 3:  # 鼠标右键点击事
            if points:
                if len(points) == 3:
                    # circle.remove()
                    ax.patches.remove(circle)
                    plt.draw()
                points.pop()
                if not points:
                    line.set_data([], [])
                else:
                    line.set_data(*zip(*points))
            else:
                line.set_data([], [])
            plt.draw()

    def draw_circle():

        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]

        # 利用三点求圆心坐标和半径
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        centerX = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (
                y1 - y2)) / D
        centerY = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (
                x2 - x1)) / D
        radius = ((x1 - centerX) ** 2 + (y1 - centerY) ** 2) ** 0.5

        circle = plt.Circle((centerX, centerY), radius, color='b', fill=False)
        circle_info.append([round(centerX, 1), round(centerY, 1), round(radius, 1)])
        ax = plt.gca()
        ax.add_patch(circle)
        plt.draw()
        return circle
    targets = {}  # List to store active targets
    target_counter = 0
    points = []
    num_threads = 16  # Adjust the number of threads as needed
    # load file
    # root = tk.Tk()
    # root.withdraw()
    # path = filedialog.askopenfilename()
    old_name = False
    csv_path = os.path.splitext(path)[0] + '.csv'
    if (os.path.exists(csv_path)):
        print('clean csv with same name')
        os.remove(csv_path)

    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    frame_number = video.get(7)
    date = datetime.date.today()

    # # 需要修改的值
    show = show_value  # show or not
    P1 = P1
    P2 = P2
    min_contourArea = min_wormArea
    max_contourArea = max_wormArea
    jump_frame = track_jump  # 追踪的跳帧

    # # 不需要修改
    skip_jump_frame = sum_jump  # 叠底的跳帧
    maximum_distance = jump_frame * 3  # 允许移动的最大距离
    if maximum_distance < 25:
        maximum_distance = 25
    sum_track = 0
    max_missing_frames = 1  # Maximum number of consecutive frames a target can be missing before it's considered disappeared

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # file_img_ls = []
    # current_cenroid_ls = []
    # create_ls = True
    # img_idx_track = 0
    print(path)
    ok, frame = video.read()
    gray = frame
    # frame_gray = gray + gray
    # frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots()
    ax.set_xlim(0, gray.shape[1])  # Set the x-axis limits based on image width
    ax.set_ylim(gray.shape[0], 0)
    line = Line2D([], [], marker='o', color='r', linestyle='None')
    ax.add_line(line)
    ax.imshow(gray)
    ax.set_aspect('auto')  # 保持图片原始比例
    plt.connect('button_press_event', on_click)
    plt.show()

    new_sheet = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
    x, y, r = circle_info[0]

    cv2.circle(new_sheet, (int(x), int(y)), int(r), 255, -1, cv2.LINE_AA)


    # frame_without_back = cv2.bitwise_and(img, new_sheet)
    # # cv2.namedWindow('Canny_img',cv2.WINDOW_NORMAL)
    # # cv2.imshow('Canny_img', frame_gray)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # #

    h, w = gray.shape[:2]
    image_diagonal_length = math.sqrt(w ** 2 + h ** 2)
    kernel_close = np.ones((3, 3), np.uint8)
    if not (os.path.exists(str(path).split('.avi')[0] + '.jpeg')):  # 判断是否已经有对应文件的叠底图片
        print('sum_track does not exist')
        sum_track = multithreaded_frame_processing(path, frame_number, num_threads, skip_jump_frame, sum_P1,
                                                   sum_P2, kernel, kernel_close, new_sheet)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sum_track, connectivity=8)
        com_ls = []
        t_len_componentMask = 0

        # 筛选面积大于10000的保留
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if 5000 < area:
                com_ls.append(i)
        for i in com_ls:
            componentMask = (labels == i).astype("uint8") * 255
            t_len_componentMask = t_len_componentMask + componentMask
        cv2.imwrite((str(path)).split('.avi')[0] + '.jpeg', t_len_componentMask)
    else:
        print('sum_track exist do next...')
    t_len_componentMask = cv2.imread((str(path)).split('.avi')[0] + '.jpeg')
    componentMask = cv2.cvtColor(t_len_componentMask, cv2.COLOR_BGR2GRAY)
    # Track Worm
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = video.read()
    # new_sheet = new_sheet.astype(np.uint8)

    init = False

    with tqdm(total=frame_number) as pbar:
        sum_track = 0
        img_idx_detect = 0
        track_t = 0
        target_ls = []
        ite = 0
        timeC = 0
        frame = ""
        gray = ""
        Canny_img = ""
        image1 = ""
        im_floodfill = ""
        mask = ""
        im_floodfill_inv = ""
        im_out = ""
        frame_without_back = ""
        filtered_image = ""
        object_sizes = ""
        track_end = False
        while ok:
            timeC += 1
            ite += 1
            if (timeC % jump_frame != 0):
                ok = video.grab()
                continue

            # for j in range(0, int(frame_number), jump_frame):ko
            ok, frame = video.read()
            if not ok:
                break
            img_idx_detect += jump_frame
            frame_gray = frame[:, :, 2]
            # frame.release()

            # frame_gray += frame_gray
            # frame_gray = np.interp(frame_gray, (100, 150), (0, 255)).astype(np.uint8)
            # Canny_img = cv2.Canny(frame_gray, P1, P2)
            # frame_gray = None
            frame_without_back = cv2.bitwise_and(frame_gray, new_sheet)
            # Canny_img = None
            Canny_img = cv2.Canny(frame_without_back, P1, P2)
            Canny_img = cv2.bitwise_and(Canny_img, componentMask)
            image1 = cv2.dilate(Canny_img, kernel, iterations=1)
            im_out = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, kernel_close)
            image1 = None
            im_floodfill = None
            im_floodfill_inv = None

            _, labels, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=8)
            filtered_image = np.zeros_like(im_out)
            im_out = None
            frame_without_back = None
            object_sizes = stats[:, cv2.CC_STAT_AREA]
            del stats
            # 筛选面积符合worm大小的objects
            # selected_labels = np.where(((object_sizes > min_contourArea) & (object_sizes < max_contourArea)))
            selected_labels = np.logical_and(object_sizes > min_contourArea, object_sizes < max_contourArea)
            track_t += jump_frame
            pbar.update(jump_frame)
            contour_ls = []
            target_len = 0
            for target in list(targets.values()):
                if target.active:
                    target.increase_missing_frames()
                    target_len += 1
            # 存储符合条件的中心点
            large_area_centroids = []

            # 获取这些对象的中心点坐标
            large_area_centroids = centroids[selected_labels]
            del centroids
            large_area_centroids_tuple = [tuple(coord) for coord in large_area_centroids]
            contour_ls = [(round(x), round(y)) for x, y in large_area_centroids_tuple]

            if len(contour_ls) == 0:
                continue
            if target_len == 0:
                init = False
            if not init:
                # if ite ==0:
                #     continue
                if len(contour_ls) == 0:
                    continue
                for new_target in contour_ls:
                    targets[target_counter] = Target(target_counter, new_target, track_t)
                    target_counter += 1
                    init = True
                continue

            last_position_ls = []
            target_ls = []
            for target in targets.values():
                if target.active:
                    # last_position = target.positions[-1][0]
                    last_position_ls.append(list(target.positions[-1][0]))
                    target_ls.append(target)

            dist_matrix = distance_matrix(np.array(last_position_ls), np.array(contour_ls), p=2)
            # 将大于maximum_distance的值设为image_diagonal_length不参与匈牙利算法配对
            dist_matrix[dist_matrix > maximum_distance] = image_diagonal_length
            # 将列值为image_diagonal_length的列从dist_matrix中删除
            inf_cols = np.where(np.all(dist_matrix == image_diagonal_length, axis=0))[0]
            ist_matrix = np.delete(dist_matrix, inf_cols, axis=1)
            del dist_matrix
            inf_raws = np.where(np.all(ist_matrix == image_diagonal_length, axis=1))[0]
            ist_matrix = np.delete(ist_matrix, inf_raws, axis=0)
            inf_cols = inf_cols.tolist()
            inf_raws = inf_raws.tolist()
            for i in inf_cols:
                targets[target_counter] = Target(target_counter, contour_ls[i], track_t)
                target_counter += 1
            c = contour_ls
            contour_ls = np.delete(contour_ls, inf_cols, axis=0)
            contour_ls = [tuple(row) for row in contour_ls]
            last_position_ls = np.delete(last_position_ls, inf_raws, axis=0)
            last_position_ls = [tuple(row) for row in last_position_ls]
            target_ls = list(np.delete(target_ls, inf_raws, axis=0))
            # contour_ls=np.array(contour_ls)
            row_indices, col_indices = linear_sum_assignment(ist_matrix)

            pair_result = []
            new_worm_pos = []
            pair_result = list(zip(row_indices, col_indices))
            # 修改新虫不追踪情况
            new_worm_idx_ls = list(range(1, len(contour_ls)))
            new_worm_idx_ls = set(new_worm_idx_ls).difference(set(col_indices))
            for worm_index, ob_element in pair_result:
                if (ist_matrix[worm_index, ob_element]) <= maximum_distance:
                    target_ls[worm_index].active = True
                    target_ls[worm_index].update_position(contour_ls[ob_element], track_t)
                else:
                    targets[target_counter] = Target(target_counter, contour_ls[ob_element], track_t)
                    target_counter += 1

            for target in list(targets.values()):
                target.check_death()
            if ite % 500 == 0:
                write_csv(track_end, csv_path, targets)
            for new_worm_idx in new_worm_idx_ls:
                targets[target_counter] = Target(target_counter, contour_ls[new_worm_idx], track_t)
                target_counter += 1

            if show:
                result = np.isin(labels, selected_labels)
                filtered_image[result] = 255
                # filtered_image = cv2.bitwise_not(filtered_image)
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
                for target in targets.values():
                    # print(target.tid)
                    if target.active:
                        x, y = target.positions[-1][0]
                        cv2.putText(filtered_image, f"ID: {target.tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)
                        cv2.putText(filtered_image, f"frame: {ite}", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 7)
                        # Draw the trajectory of the target
                        for i in range(1, len(target.positions)):
                            prev_x, prev_y = target.positions[i - 1][0]
                            curr_x, curr_y = target.positions[i][0]
                            cv2.line(filtered_image, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 2)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                fil = filtered_image
                cv2.imshow('image', filtered_image)
                # cv2.imshow('image', fil[750:950,1000:1200])
                cv2.waitKey(80)

        # Save data
        track_end = True
        write_csv(track_end, csv_path, targets)
        with open((str(path)).split('.avi')[0] + '_circle_info.csv', 'w', newline='') as circle_csvfile:
            csv_writer = csv.writer(circle_csvfile)
            csv_writer.writerow(['circle_X', 'circle_Y', 'Radius'])
            for circle in circle_info:
                circle_X, circle_y, radius = circle
                csv_writer.writerow([circle_X, circle_y, radius])
        circle_csvfile.close()


def write_csv(track_end, csv_path, targets):
    del_id = []

    # with open(csv_path, 'w', newline='') as csvfile:
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csv_writer.writerow(['ID', 'X', 'Y', 'Timestamp'])
        for target_id, target in list(targets.items()):
            if not track_end:
                if not target.active:
                    for position, timestamp in target.positions:
                        x, y = position
                        csv_writer.writerow([target_id, x, y, timestamp])
                    del_id.append(target_id)

                if target.active:
                    if len(target.positions) > 2:
                        positions_to_write = target.positions[:-2]
                        # 保留最后两个数据
                        positions_to_keep = target.positions[-2:]
                        # 写入除后两个外的其它数据
                        for position, timestamp in positions_to_write:
                            x, y = position
                            csv_writer.writerow([target_id, x, y, timestamp])
                        # 保留最后两个数据
                        target.positions = positions_to_keep
                    else:
                        continue
            # track 结束
            else:
                for position, timestamp in target.positions:
                    x, y = position
                    csv_writer.writerow([target_id, x, y, timestamp])
                del_id.append(target_id)
        if track_end:
            csvfile.close()
        for idx in del_id:
            del targets[idx]
