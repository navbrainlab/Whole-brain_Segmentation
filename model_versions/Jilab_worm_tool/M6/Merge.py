import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.animation as animation
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler  #
from operator import itemgetter
import tkinter as tk
from tkinter import filedialog
import csv
import math
import os
import time
from datetime import datetime
import copy


# from tkinter import _flatten
# 手动调参
def merge(root, noise_clear_area, cluster_number_1, cluster_number_2, time_threshold, time_number, max_time,
          max_distance, area_threshold, typical_diagonal_Length_threshold, noise_clear_area_2, show_value):


    root.destroy()
    show = show_value
    # 第一次降噪
    # noise_clear_area = 200  # 此参数是用来筛取对角线面积较小的轨迹并认为它们是噪声，如果觉得筛选强度太厉害，请把此值调大
    c = 3  # 此参数用于检索噪音点是否集中在一片较小区域，一般不需更改
    min_sample_of_cluster = 5  # 此参数用于检索噪音点是否聚集在一起，若此参数为n，即至少有n个轨迹聚集在小区域中。不建议设得太小
    # cluster_number_1 = 15  # 此参数聚集量越大，其是闪烁噪音的可能性就越大。可以调大
    # cluster_number_2 = 5
    # time_threshold = 200000  # 除长时噪声

    # 合成
    # time_number = 40  # 合成步骤的迭代次数。迭代次数越多，作为是否能合成判据的半径和时间阈值会越高
    # max_time = 2  # 时间阈值的初始值。不建议设得太大
    # max_distance = 2.2  # 6cm（2.2) 9cm(2）

    # 二次降噪

    # distance_threshold = 30 #二次降噪的判据是轨迹的矩形对角线长度，因此，如果阈值越小，会有更多轨迹被保留，但也可能会保留更多细碎轨迹
    # area_threshold = 10

    # 删除边缘小轨迹
    size = 9  # 这里是几cm的盘子
    # center_point = (2010.3, 1456.5) #此处定义的是数据集圆心坐标，对不同的数据集，应该在开始前手动输入
    distance_to_center = 1500  # 此处定义了轨迹起、终点与圆心的位置，若超出这个数值，说明轨迹整体离圆心太远，如果筛选强度过大，请将此值调大
    # typical_diagonal_Length_threshold = 5000  # 此处为应当被筛除的太小轨迹定义了阈值，小于这个阈值，说明跨度太小，作为边缘小轨迹去除。如果筛选强度过大，请将此值调小
    size_to_pixel = {9: [0.18], 6: [0.2]}

    # 第二次dbscan使用的参数
    # noise_clear_area_2 = 500

    # 用于画后来的柱状图
    name_list = ['original', '1st noise_cleared', 'after merge', '2st noise_cleared', 'little track']

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory(title='请选择一个路径文件（包含所有要merge文件名）')
    # circle_path = filedialog.askdirectory(title='请选择圆心信息所在路径文件（包含所有要merge文件名）')
    # date= datetime.date.today()
    data_files = sorted([f for f in os.listdir(file_path) if f.endswith(('11111.csv'))])
    # circle_files = sorted([f for f in os.listdir(file_path)if f.endswith(('_circle_info.csv') )])
    nsy_path = r'Z:\nsy\collision_dataspace'  # 把碰撞数据放到这里来
    print(data_files)
    print('共' + str(len(data_files)) + '个文件')
    # zip()
    merge_num = 0
    for i in data_files:
        # 是否进行碰撞检测
        collision_detection = 0

        file_name = i
        circle_name = str(i[:-4]) + '_circle_info.csv'
        center_data_df = pd.read_csv(str(file_path) + '/' + circle_name)
        center_point = [center_data_df['circle_X'][0], center_data_df['circle_Y'][0]]
        radius = center_data_df['Radius'][0]
        distance_to_center = radius - 10
        print(center_point)
        print(radius)
        id_list = []
        print('正在merge的文件： ' + str(i[:-4]))

        nsy_save_path = str(nsy_path) + '/' + str(i[:-4])

        save_path = str(file_path) + '/' + str(i[:-4]) + '_after_merge_1030_14'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # t_1=time.time()
        # 增加参数
        print(str(file_path) + '/' + i)
        # print(str(file_path)+'/'+i)
        df_raw = pd.read_csv(str(file_path) + '/' + i)
        df = df_raw.copy()

        ###########################################################
        # 更新一下misalignment_area和misalignment_rate
        misalignment_area = {}
        misalignment_rate = {}

        for group_id, group_data in df.groupby('ID'):
            positions = group_data.apply(lambda row: (row['X'], row['Y']), axis=1)
            misalignment_area[group_id] = len(set(list(positions)))
            misalignment_rate[group_id] = (len(set(list(positions)))) / len(positions)

        df['Misalignment_area'] = df['ID'].map(misalignment_area)
        df['Misalignment_rate'] = df['ID'].map(misalignment_rate)
        #############################################################

        count = 0
        id_data = {}
        diagonal_lengths_rate = {}
        diagonal_lengths = {}
        rect_centers = {}
        rect_area = {}

        ###################################################################
        plt.figure()
        for group_id, group_data in df.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            timestamp = group_data['Timestamp']
            time_diff = timestamp.iloc[-1] - timestamp.iloc[0]
            # print(group_id)
            # plt.plot(x, y)

            #     # start_time=timestamp.iloc[0]
            #     # end_time=timestamp.iloc[-1]
            # #     plt.text(x.iloc[0],y.iloc[0],str(group_id)+":s"+str(start_time))
            # #     plt.text(x.iloc[-1],y.iloc[-1],str(group_id)+":e"+str(end_time))
            # #     plt.text(x.iloc[-1],y.iloc[-1],str(group_id))
            points = list(zip(x, y))
            # print(points)
            points_np = np.array(points)
            rect = cv2.minAreaRect(points_np)
            center, size, angle = rect
            width, height = size
            time_diff = timestamp.iloc[-1] - timestamp.iloc[0]
            if time_diff == 0:
                diagonal_lengths[group_id] = np.nan
                diagonal_lengths_rate[group_id] = 9999
                rect_centers[group_id] = np.nan
                rect_area[group_id] = np.nan
                # df = df.drop(df[df["ID"] == group_id].index)
            else:
                diagonal_lengths[group_id] = round((np.sqrt(width ** 2 + height ** 2)), 2)
                diagonal_lengths_rate[group_id] = round((np.sqrt(width ** 2 + height ** 2)) / time_diff, 2)
                rect_centers[group_id] = center
                rect_area[group_id] = round(width, 2) * round(height, 2)
                area = rect_area[group_id]
                if area < noise_clear_area and area != np.nan:
                    # if len(rect_centers[group_id])== 1:
                    #     print(rect_centers[group_id])
                    id_data[group_id] = rect_centers[group_id]

        if show:
            plt.show()
        id_list.append(len(df['ID'].unique()))

        df['Diagonal_Length_rate'] = df['ID'].map(diagonal_lengths_rate)
        df['Diagonal_Length'] = df['ID'].map(diagonal_lengths)
        df['rect_center'] = df['ID'].map(rect_centers)
        df['rect_area'] = df['ID'].map(rect_area)

        df = df.drop(df[df["Diagonal_Length_rate"] == 9999].index)

        # merge算法
        def merge_two_track(TIMES, df_clear, max_time, max_distance):
            #     df_clear=df_after_clear
            df_clear_merge_final = df_clear.copy()
            max_time = max_time * TIMES
            max_distance = max_distance * TIMES

            start_time = df_clear.groupby('ID').first().reset_index()
            end_time = df_clear.groupby('ID').last().reset_index()

            end_time_matrix = np.tile(end_time['Timestamp'], (len(start_time['Timestamp']), 1))
            start_time_matrix = np.tile(start_time['Timestamp'], (len(end_time['Timestamp']), 1)).T
            time_distance_matrix = (start_time_matrix - end_time_matrix)
            row_indices, col_indices = np.where((time_distance_matrix < max_time) & (time_distance_matrix > 0))

            if len(row_indices) == 0:
                return df_clear, 0
            end_time_intime = end_time.iloc[col_indices, :].reset_index(drop=True)
            start_time_intime = start_time.iloc[row_indices, :].reset_index(drop=True)
            start_time_intime['Coordinates'] = start_time_intime.apply(lambda row: (row['X'], row['Y']), axis=1)
            end_time_intime['Coordinates'] = end_time_intime.apply(lambda row: (row['X'], row['Y']), axis=1)
            pos_dist_matrix = distance.cdist(list(end_time_intime['Coordinates']),
                                             list(start_time_intime['Coordinates']),
                                             metric='euclidean')
            diagonal_indices = np.where(np.diag(pos_dist_matrix) < max_distance)[0]
            end_start_intimepos = pd.DataFrame(
                {"end_id": (end_time_intime.loc[diagonal_indices, 'ID']).reset_index(drop=True),
                 "start_id": (start_time_intime.loc[diagonal_indices, 'ID']).reset_index(drop=True)
                 })
            end_start_intimepos = end_start_intimepos[end_start_intimepos["end_id"] != end_start_intimepos["start_id"]]
            end_start_intimepos.reset_index(drop=True)

            unique_values, value_counts = np.unique(end_start_intimepos['start_id'], return_counts=True)
            values_start_greater_than_2 = unique_values[value_counts >= 2].tolist()
            values_start_eq1 = unique_values[value_counts == 1].tolist()

            unique_values, value_counts = np.unique(end_start_intimepos['end_id'], return_counts=True)
            values_end_greater_than_2 = unique_values[value_counts >= 2].tolist()
            values_end_eq1 = unique_values[value_counts == 1].tolist()
            collision = []
            collision = values_end_greater_than_2
            filtered_collision_end_rows = end_start_intimepos[
                end_start_intimepos['end_id'].isin(values_end_greater_than_2)]
            filtered_collision_start_rows = end_start_intimepos[
                end_start_intimepos['start_id'].isin(values_start_greater_than_2)]
            filtered_collision_rows = pd.concat([filtered_collision_end_rows, filtered_collision_start_rows])
            #     filtered_collision_rows = filtered_collision_rows[filtered_collision_rows["end_id"] != filtered_collision_rows["start_id"]]
            filtered_collision_rows = filtered_collision_rows.drop_duplicates(keep='first')
            filtered_collision_rows = filtered_collision_rows.sort_values(by=['end_id'])

            filtered_end_rows = end_start_intimepos[end_start_intimepos['end_id'].isin(values_end_eq1)]
            filtered_rows = filtered_end_rows[filtered_end_rows['start_id'].isin(values_start_eq1)]
            filtered_rows.reset_index(drop=True)
            merge_ID_ls = filtered_rows.reset_index(drop=True)
            merge_ls = []
            # 连续找出需要合成的ID
            while 1:
                if len(merge_ID_ls) == 0:
                    break
                a = 0
                start_id = merge_ID_ls.start_id
                end_id = merge_ID_ls.end_id
                merge_target = [end_id[a], start_id[a]]
                drop_ls = []
                while 1:
                    drop_ls.append(a)
                    idx = end_id[end_id == start_id[a]].index.tolist()
                    if len(idx) == 0:
                        break
                    merge_target.append(start_id[idx[0]])
                    a = idx[0]
                merge_ID_ls = merge_ID_ls.drop(drop_ls)
                merge_ID_ls = merge_ID_ls.reset_index(drop=True)
                merge_ls.append(merge_target)
            # merge ID
            for i in merge_ls:
                for j in i:
                    df_clear_merge_final.loc[df_clear_merge_final['ID'] == j, 'ID'] = min(i)

            # print(merge_ls)
            for end in filtered_collision_rows['end_id'].unique().tolist():
                if end in values_start_eq1:
                    for item in merge_ls:
                        if end in (item):
                            filtered_collision_rows.loc[filtered_collision_rows['end_id'] == end, 'end_id'] = min(item)

            for start in filtered_collision_rows['start_id'].unique().tolist():
                if start in values_end_eq1:
                    for item in merge_ls:
                        if start in (item):
                            filtered_collision_rows.loc[filtered_collision_rows['start_id'] == start, 'start_id'] = min(
                                item)

            return df_clear_merge_final, filtered_collision_rows

        # merge前数据
        count = 0

        # 删除内部小噪音
        from sklearn.cluster import DBSCAN

        df_clear = df.copy()

        # 划分区域
        # 以质心位置的x坐标最小、最大值作为划分标准
        rect_center_x_min = min(np.array(list(id_data.values()))[:, 0])
        rect_center_x_max = max(np.array(list(id_data.values()))[:, 0])

        layer_number = 50  # 把空间分割成n份

        for i in range(layer_number):
            x_range_min = rect_center_x_min + ((rect_center_x_max - rect_center_x_min) * i) / layer_number
            x_range_max = rect_center_x_min + ((rect_center_x_max - rect_center_x_min) * (i + 1)) / layer_number

            X = np.array(
                [point for point in list(id_data.values()) if (point[0] < x_range_max and point[0] > x_range_min)])

            # X = np.array(list(id_data.values()))

            if len(X.shape) == 1:
                # print("第一次除噪未发现噪音")
                continue
            else:

                clusters = DBSCAN(eps=cluster_radius, min_samples=cluster_number_1).fit_predict(X)
                # 对噪音的中心进行聚类
                index_clusters = {}

                # 输出为正常的轨迹id与聚类id的dataframe

                should_delete = []
                cluster_array = np.array(clusters)

                id_array = np.array([id for id in list(id_data.keys()) if
                                     (id_data[id][0] < x_range_max and id_data[id][0] > x_range_min)])

                cluster_array[cluster_array != -1] = 1

                id_cluster = np.multiply(cluster_array, id_array)
                new_id = id_cluster[id_cluster >= 0]

                # 如果聚类数大于某个阈值，那么就认为那是一个闪烁的噪音，把它的id加入到一个列表中

                df_clear = df_clear[-df_clear["ID"].isin(new_id)]
                # 删除所有噪音点后，得到的就是去除噪音的数据

        plt.figure()
        for group_id, group_data in df_clear.groupby('ID'):
            x_track = group_data['X']
            y_track = group_data['Y']
            timestamp = group_data['Timestamp']
            #     plt.text(x_track.iloc[-1],y_track.iloc[-1],str(group_id))
            plt.plot(x_track, y_track)
        plt.title('tracks that noise cleared -ID num:' + str(len(df_clear['ID'].unique())))
        plt.savefig(save_path + '/' + name_list[1] + '.png', bbox_inches='tight')
        if show:
            plt.show()
        id_list.append(len(df_clear['ID'].unique()))

        ##################################################################################################################
        df_select = df_clear
        # df_clear_short = df_clear[df_clear['ID'] < 0]
        # print("qqq")
        # df_clear_short = df_clear[df_clear['ID'] < 0]
        # df_clear_short.loc[df_clear_short['ID']==group_id,'time_phase'] = time_phase
        need_id = []
        # time_threshold = (max(df_clear['Timestamp'].unique())-min(df_clear['Timestamp'].unique()))/1000

        for group_id, group_data in df_clear.groupby('ID'):
            # 按id取起点，终点
            #     df_clear_short = pd.concat([df_clear_short,df_clear[df_clear['ID']==group_id].iloc[[0]]])
            #     df_clear_short_1 = pd.concat([df_clear_short,df_clear[df_clear['ID']==group_id].iloc[[-1]]])
            #     print(df_clear['Timestamp'][df_clear['ID']==group_id].iloc[-1])
            time_phase = df_clear['Timestamp'][df_clear['ID'] == group_id].iloc[-1] - \
                         df_clear['Timestamp'][df_clear['ID'] == group_id].iloc[0]

            if time_phase > time_threshold and df_clear['rect_area'][df_clear['ID'] == group_id].iloc[0] < 10:
                need_id.append(group_id)
        #     print("aaa")
        #     time_phase_list.append(time_phase)
        # df_clear_short.loc[:,'time_gap']=df_clear_short_1['Timestamp']-df_clear_short['Timestamp']
        # df_time=pd.DataFrame({'time_gap':time_phase_list})
        # print("aaa")
        df_clear = df_clear[-df_clear['ID'].isin(need_id)]
        # df_select

        # ######################################################################################################################
        # 删除外周的小轨迹

        # df_clear_short = df_select[df_select['ID'] < 0]
        # 计算每一点与原点的距离

        # for group_id ,group_data in df_clear.groupby('ID'):
        #
        #     # print(df_clear[df_clear['ID']==group_id].iloc[0])
        #     df_clear_short = pd.concat([df_clear_short,df_clear[df_clear['ID']==group_id].iloc[[0,-1]]])

        # 按id取起点，终点
        df_clear_short = pd.concat(
            [df_clear.groupby('ID', as_index=False).first(), df_clear.groupby('ID', as_index=False).last()])
        df_clear_short = df_clear_short.sort_values(by='ID', ascending=True)

        df_clear_short['Coordinates'] = df_clear_short.apply(lambda row: (int(row['X']), int(row['Y'])), axis=1)

        center_point_matrix = [center_point, center_point]

        dist_matrix = distance.cdist(list(df_clear_short['Coordinates']), center_point_matrix)
        df_clear_short['d_2_c'] = dist_matrix[:, 0]

        # for i in df_clear_short.index:
        #     df_clear_short.loc[i, 'd_2_c'] = np.sqrt((np.array(df_clear_short['Coordinates'][df_clear_short.index == i])[0][0]-center_point[0])**2 + (np.array(df_clear_short['Coordinates'][df_clear_short.index == i])[0][1]-center_point[1])**2)

        id_data = {}

        start_point_to_center = df_clear_short.groupby('ID').first()['d_2_c'].tolist()
        end_point_to_center = df_clear_short.groupby('ID').last()['d_2_c'].tolist()
        diagonal_length = df_clear_short.groupby('ID').last()['Diagonal_Length'].tolist()
        id_list_1 = df_clear_short['ID'].unique().tolist()
        id_data = pd.DataFrame([id_list_1, start_point_to_center, end_point_to_center, diagonal_length],
                               index=['ID', 'start_2_center', 'end_2_center', 'Diagonal_Length'])
        id_data = id_data.T

        # for id in df_clear_short['ID'].unique():
        #     present_data = []
        #     present_data.append(np.array(df_clear_short.loc[df_clear_short['ID'] == id, ['d_2_c']])[0][0])  # 传入起点与圆心距离
        #     present_data.append(np.array(df_clear_short.loc[df_clear_short['ID'] == id, ['d_2_c']])[-1][0])  # 传入终点坐标与圆心距离
        #     present_data.append(
        #         np.array(df_clear_short.loc[df_clear_short['ID'] == id, ['Diagonal_Length']])[0][0])  # 传入外接矩形对角线长度
        #     id_data[id] = present_data

        suspect_ids = []
        diagonal_Length_threshold = typical_diagonal_Length_threshold * 0.18

        suspect_data = id_data.loc[id_data['start_2_center'] > distance_to_center]
        suspect_data = suspect_data.loc[suspect_data['end_2_center'] > distance_to_center]
        suspect_data.loc[suspect_data['Diagonal_Length'] < diagonal_Length_threshold]

        suspect_ids = suspect_data['ID'].tolist()

        # for i in id_data.keys():
        #     if (id_data[i][0] > distance_to_center and id_data[i][1] > distance_to_center) and id_data[i][2] < diagonal_Length_threshold:
        #         suspect_ids.append(i)
        # # print(suspect_ids)

        df_clear_done = df_clear.copy()
        # 暂时去掉小轨迹们
        df_clear_done = df_clear_done[-df_clear_done['ID'].isin(suspect_ids)]

        # 再创建一个数据框，用来存储暂时被舍弃的小轨迹
        df_little_track = df_clear.copy()
        df_little_track = df_little_track[df_little_track['ID'].isin(suspect_ids)]

        plt.figure()
        for group_id, group_data in df_little_track.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            # time = group_data['Timestamp']
            plt.plot(x, y)
            # plt.text(x.iloc[0], y.iloc[0], str(group_id))
            # plt.text(x.iloc[-1], y.iloc[-1], str(group_id))
        plt.title('cleared margin tracks-ID num:' + str(len(df_clear_done['ID'].unique())))
        plt.savefig(save_path + '/' + name_list[4] + '.png', bbox_inches='tight')
        # plt.savefig(save_path + '/' + name_list[2] + '.png', bbox_inches='tight')
        if show:
            plt.show()
        # ##########################################################################################################################

        # 初级筛选以及merge
        # df=df_clear
        # df_clear_twice=df[(df['Misalignment_rate']>0.05) | (df['Diagonal_Length_rate']>0.05)]
        df_merge = df_clear_done
        count = 0
        for i in range(1, time_number):
            df_merge, collision_show = merge_two_track(i, df_merge, max_time, max_distance)

        # collision_show.to_csv(save_path + '/' + 'k_1.csv')

        ########################################################

        # for item in id_vector_enter

        # print(id_vector)

        # update parameters

        df_merge_update = df_merge.copy()
        diagonal_lengths_rate_update = {}
        diagonal_lengths_update = {}
        Misalignment_area_update = {}
        Misalignment_rate_update = {}
        rect_center_update = {}
        rect_area_update = {}
        plt.figure()
        for group_id, group_data in df_merge_update.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            time_1 = group_data['Timestamp']
            start_time = time_1.iloc[0]
            end_time = time_1.iloc[-1]
            #     plt.text(x.iloc[0],y.iloc[0],str(group_id)+":s"+str(start_time))
            #     plt.text(x.iloc[-1],y.iloc[-1],str(group_id)+":e"+str(end_time))
            plt.title('tracks after merge-ID num:' + str(len(df_merge_update['ID'].unique())))
            plt.plot(x, y)

            points = list(zip(x, y))
            points_np = np.array(points)
            rect = cv2.minAreaRect(points_np)
            center, size, angle = rect
            width, height = size
            time_diff = time_1.iloc[-1] - time_1.iloc[0]
            Misalignment_rate_update[group_id] = round((len(set(list(map(itemgetter(0), points))))) / time_diff, 2)
            Misalignment_area_update[group_id] = len(set(list(map(itemgetter(0), points))))
            diagonal_lengths_update[group_id] = round((np.sqrt(width ** 2 + height ** 2)), 2)
            diagonal_lengths_rate_update[group_id] = round((np.sqrt(width ** 2 + height ** 2)) / time_diff, 2)
            rect_center_update[group_id] = center
            rect_area_update[group_id] = width * height
        df_merge_update['Misalignment_area'] = df_merge_update['ID'].map(Misalignment_area_update)
        df_merge_update['Misalignment_rate'] = df_merge_update['ID'].map(Misalignment_rate_update)
        df_merge_update['Diagonal_Length'] = df_merge_update['ID'].map(diagonal_lengths_update)
        df_merge_update['Diagonal_Length_rate'] = df_merge_update['ID'].map(diagonal_lengths_rate_update)
        df_merge_update['rect_center'] = df_merge_update['ID'].map(rect_center_update)
        df_merge_update['rect_area'] = df_merge_update['ID'].map(rect_area_update)
        plt.savefig(save_path + '/' + name_list[2] + '.png', bbox_inches='tight')
        if show:
            plt.show()
        id_list.append(len(df_merge_update['ID'].unique()))
        # df_merge_update.to_csv(save_path + '/' + 'merge.csv')
        # 将删掉的小轨迹重新加入到数据中

        # 进一步筛除噪声
        id_data = {}
        df_2nd_dbscan = df_merge_update.copy()
        ###################################################################
        for group_id, group_data in df_2nd_dbscan.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            timestamp = group_data['Timestamp']
            time_diff = timestamp.iloc[-1] - timestamp.iloc[0]

            points = list(zip(x, y))
            if time_diff == 0:
                diagonal_lengths[group_id] = np.nan
                diagonal_lengths_rate[group_id] = 9999
                rect_centers[group_id] = np.nan
                rect_area[group_id] = np.nan
            else:
                area = rect_area[group_id]
                if area < noise_clear_area_2 and area != np.nan:
                    id_data[group_id] = rect_centers[group_id]

        X = np.array(list(id_data.values()))

        if len(X.shape) == 1:
            print("第一次除噪未发现噪音")
        else:

            clusters = DBSCAN(eps=cluster_radius, min_samples=cluster_number_2).fit_predict(X)
            # 对噪音的中心进行聚类
            index_clusters = {}

            # 输出为正常的轨迹id与聚类id的dataframe

            should_delete = []
            cluster_array = np.array(clusters)

            id_array = np.array(list(id_data.keys()))

            cluster_array[cluster_array != -1] = 1

            id_cluster = np.multiply(cluster_array, id_array)
            new_id = id_cluster[id_cluster >= 0]

            # 如果聚类数大于某个阈值，那么就认为那是一个闪烁的噪音，把它的id加入到一个列表中

            df_2nd_dbscan = df_2nd_dbscan[-df_2nd_dbscan["ID"].isin(new_id)]
            # 删除所有噪音点后，得到的就是去除噪音的数据
        # df_2nd_dbscan.to_csv(save_path + '/' + 'df_2nd_dbscan.csv')

        df_after_second_clear = df_2nd_dbscan[(df_2nd_dbscan['Misalignment_area'] > area_threshold)]
        # a = df_after_second_clear[df_after_second_clear['ID'] == 598]
        ############################################
        count = 0
        time_number = 80
        for i in range(1, time_number):
            # print(i)
            df_after_second_clear, collision_show = merge_two_track(i, df_after_second_clear, max_time, max_distance)
            # a = df_after_second_clear[df_after_second_clear['ID']==598]
        if type(collision_show) != int:
            collision_show.to_csv(nsy_save_path + '_collision_show.csv')
            df_after_second_clear.to_csv(nsy_save_path + '_merged_data.csv')
            ############################################

            k = collision_show
            collision_detection = 0
            # 加入碰撞检测及碰撞对merge
            if k.shape[0] == 0:
                collision_detection = 1
                print('没有碰撞！')
            if collision_detection == 0:
                # 从数据中提取碰撞点
                # 以元组形式组成：collision_group=[[[1,2],[3,4]],[[5,6],[7,8]],...]
                # 其中，1，2，3，4组成一个碰撞组。1，2是入，3，4是出；以此类推
                start_id_list = {}

                for group_id, group_data in k.groupby('start_id'):

                    if k[k['start_id'] == group_id].shape[0] == 2:
                        start_id_list[group_id] = list(group_data['end_id'])

                collision_group = {}
                corrent_group = []
                corrent_pair = []

                all_pairs = []
                end_id_group = []
                for item in list(start_id_list.values()):
                    if item in end_id_group:
                        pass
                    else:
                        end_id_group.append(item)
                # print(end_id_group)
                for item in end_id_group:
                    for i in start_id_list.keys():
                        if start_id_list[i][0] == item[0] and start_id_list[i][1] == item[1]:
                            corrent_group.append(i)
                        else:
                            continue
                    corrent_pair.append(item)
                    corrent_pair.append(corrent_group)
                    if len(corrent_group) == 2:
                        all_pairs.append(corrent_pair)
                    corrent_group = []
                    corrent_pair = []
                if all_pairs == []:
                    collision_detection = 1
                    print('没有可以鉴别的碰撞！')
                else:
                    # case 1：
                    # R匹配R，D匹配D即可
                    def case_1(pair, pair_judge_0):
                        paired_id = []
                        first_pair = []
                        second_pair = []
                        first_pair.append(pair[0][0])
                        second_pair.append(pair[0][1])
                        if pair_judge_0[pair[1][0]] == pair_judge_0[pair[0][0]]:
                            first_pair.append(pair[1][0])
                            second_pair.append(pair[1][1])
                        else:
                            first_pair.append(pair[1][1])
                            second_pair.append(pair[1][0])
                        paired_id.append(first_pair)
                        paired_id.append(second_pair)
                        # paired_id.append('D to R')
                        return paired_id

                    # case 2：
                    # 计算质心距离矩阵，取距离最短的匹配。
                    def case_2(df_enter, df_exit, pair):
                        paired_id = []
                        first_pair = []
                        second_pair = []
                        # print(pair[0][0])
                        # print(df_enter['ID'])
                        enter_center_pts = [df_enter[df_enter['ID'] == pair[0][0]]['center'].tolist()[0],
                                            df_enter[df_enter['ID'] == pair[0][1]]['center'].tolist()[0]]
                        after_center_pts = [df_exit[df_exit['ID'] == pair[1][0]]['center'].tolist()[0],
                                            df_exit[df_exit['ID'] == pair[1][1]]['center'].tolist()[0]]
                        # print(df_enter[df_enter['ID'] == pair[0][0]]['center'])
                        # print(enter_center_pts)
                        # print(after_center_pts)
                        dis_matrix = distance.cdist(enter_center_pts, after_center_pts, 'euclidean')
                        first_pair.append(pair[0][0])
                        second_pair.append(pair[0][1])

                        if dis_matrix[0][0] < dis_matrix[0][1]:
                            first_pair.append(pair[1][0])
                            second_pair.append(pair[1][1])
                        else:
                            first_pair.append(pair[1][1])
                            second_pair.append(pair[1][0])
                        paired_id.append(first_pair)
                        paired_id.append(second_pair)
                        # paired_id.append('D to D')
                        return paired_id

                    # case 3：
                    # 进入之前的算法
                    def case_3(pair, track_df):

                        def velocity_calculate(group, n):
                            global df_collision_enter, df_collision_exit, df_enter_calculate, df_exit_calculate
                            # df_collision：碰撞数据
                            # 定义滑动窗口。n：对每个点而言，向上回溯的点数量

                            # 定义单点速度
                            # 假设一帧为一个点，该点速度定义：
                            # （该点与回溯窗口数n后的上一点之间距离）/两点间历时
                            chase_number = n + time_before_collision
                            # 入轨迹：最后一条记录往上回溯time_before_collision + n条记录
                            # 出轨迹：第一条记录往下追溯time_before_collision + n条记录
                            # for group in all_pairs:

                            for enter_id in group[0]:
                                #             print(enter_id)

                                df_collision_enter = pd.merge(df_collision_enter,
                                                              df_collision_1[df_collision_1['ID'] == enter_id].tail(
                                                                  time_before_collision + 1), how='outer')
                                df_enter_calculate = pd.merge(df_enter_calculate,
                                                              df_collision_1[df_collision_1['ID'] == enter_id].iloc[
                                                              -(n + time_before_collision):-(n - 1)], how='outer')
                            #             print(df_collision_1[df_collision_1['ID']==enter_id].iloc[-(n+time_before_collision):-(n-1)].shape[0])
                            # 单独处理入轨迹的速度
                            df_collision_enter['Coordinates'] = df_collision_enter.apply(
                                lambda row: (row['X'], row['Y']), axis=1)
                            #         for i in range(df_collision_enter.shape[0]):
                            df_enter_calculate['Coordinates'] = df_enter_calculate.apply(
                                lambda row: (row['X'], row['Y']), axis=1)

                            # 重新设置索引，否则会根据原索引传输值，出现问题
                            df_collision_enter = df_collision_enter.reset_index(drop=True)
                            df_enter_calculate = df_enter_calculate.reset_index(drop=True)

                            df_collision_enter['last_point_Coordinates'] = df_enter_calculate['Coordinates']
                            df_collision_enter['last_point_time'] = df_enter_calculate['Timestamp']
                            #         print(df_collision_enter)

                            # 开始神圣的计算吧！
                            for i in df_collision_enter.index:
                                df_collision_enter.loc[i, 'velocity'] = math.dist(
                                    df_collision_enter.loc[i, 'last_point_Coordinates'],
                                    df_collision_enter.loc[i, 'Coordinates']) / abs(
                                    df_collision_enter.loc[i, 'Timestamp'] - int(
                                        df_collision_enter.loc[i, 'last_point_time']))

                            #         for i in  df_collision_enter.index:
                            #             print(df_collision_enter.loc[i,'last_point_Coordinates'])
                            #             print(df_collision_enter.loc[i,'Coordinates'])
                            #             print(math.dist(df_collision_enter.loc[i,'last_point_Coordinates'],
                            #                                                              df_collision_enter.loc[i,'Coordinates']))

                            #             df_collision_enter.loc[i,'velocity'] = math.dist(df_collision_enter.loc[i,'last_point_Coordinates'],
                            #                                                              df_collision_enter.loc[i,'Coordinates'])
                            # for group in all_pairs:
                            for exit_id in group[1]:
                                df_collision_exit = pd.merge(df_collision_exit,
                                                             df_collision_1[df_collision_1['ID'] == exit_id].head(
                                                                 time_before_collision + 1), how='outer')
                                df_exit_calculate = pd.merge(df_exit_calculate,
                                                             df_collision_1[df_collision_1['ID'] == exit_id].iloc[
                                                             n:n + time_before_collision + 2], how='outer')
                            #             print(df_collision_1[df_collision_1['ID']==exit_id].iloc[n:n+time_before_collision+1].shape[0])
                            #             print(df_collision_1[df_collision_1['ID']==exit_id].iloc[n:n+time_before_collision])
                            #             print("oooo")
                            df_collision_exit['Coordinates'] = df_collision_exit.apply(lambda row: (row['X'], row['Y']),
                                                                                       axis=1)
                            df_exit_calculate['Coordinates'] = df_exit_calculate.apply(lambda row: (row['X'], row['Y']),
                                                                                       axis=1)

                            # 重新设置索引，否则会根据原索引传输值，出现问题
                            df_collision_exit = df_collision_exit.reset_index(drop=True)
                            df_exit_calculate = df_exit_calculate.reset_index(drop=True)

                            df_collision_exit['last_point_Coordinates'] = df_exit_calculate['Coordinates']
                            df_collision_exit['last_point_time'] = df_exit_calculate['Timestamp']

                            for i in df_collision_exit.index:
                                df_collision_exit.loc[i, 'velocity'] = distance.euclidean(
                                    df_collision_exit.loc[i, 'last_point_Coordinates'],
                                    df_collision_exit.loc[i, 'Coordinates']) / abs(
                                    df_collision_exit.loc[i, 'Timestamp'] - int(
                                        df_collision_exit.loc[i, 'last_point_time']))

                            df_collision_enter = df_collision_enter.fillna(0.000001)
                            df_collision_exit = df_collision_exit.fillna(0.000001)
                            df_collision_enter = df_collision_enter.replace({'velocity': {0: 0.000001}})
                            df_collision_exit = df_collision_exit.replace({'velocity': {0: 0.000001}})
                            #     df_collision_exit['velocity'].replace(0,0.01)
                            #     print(df_collision_enter[df_collision_enter['velocity']==0])

                            # df_collision_enter.to_csv('C:/Users/Admin/Desktop/df_collision_enter.csv')
                            # df_enter_calculate.to_csv('C:/Users/Admin/Desktop/df_enter_calculate.csv')
                            # df_collision_exit.to_csv('C:/Users/Admin/Desktop/df_collision_exit.csv')
                            # df_exit_calculate.to_csv('C:/Users/Admin/Desktop/df_exit_calculate.csv')
                            return df_collision_enter, df_collision_exit

                        # 定义角速度计算函数
                        def angle_calculate(df_enter_angle, df_exit_angle, df_enter_calculate_angle,
                                            df_exit_calculate_angle,
                                            df_enter_calculate_real, df_exit_calculate_real, group, n):
                            # global df_enter_angle, df_exit_angle, df_enter_calculate_angle, df_exit_calculate_angle, df_enter_calculate_real, df_exit_calculate_real
                            # df_collision：碰撞数据
                            # 定义滑动窗口。n：对每个点而言，向上回溯的点数量

                            # 定义单点速度
                            # 假设一帧为一个点，该点速度定义：
                            # （该点与回溯窗口数n后的上一点之间距离）/两点间历时
                            chase_number = n + time_for_angle
                            # 入轨迹：最后一条记录往上回溯time_for_angle + n条记录
                            # 出轨迹：第一条记录往下追溯time_for_angle + n条记录
                            # for group in all_pairs:
                            # print(group)
                            # for enter_id in group[0]:
                            for enter_id in group[0]:
                                #             print(enter_id)
                                if len(df_collision_1[df_collision_1['ID'] == enter_id]) < 40:
                                    n = 1
                                    df_enter_angle = pd.merge(df_enter_angle,
                                                              df_collision_1[df_collision_1['ID'] == enter_id].tail(5),
                                                              how='outer')

                                    df_enter_calculate_angle = pd.merge(df_enter_calculate_angle,
                                                                        df_collision_1[
                                                                            df_collision_1['ID'] == enter_id].iloc[
                                                                        -(2 * (n - 1) + 5):-2 * (n - 1)], how='outer')
                                    df_enter_calculate_real = pd.merge(df_enter_calculate_real,
                                                                       df_collision_1[
                                                                           df_collision_1['ID'] == enter_id].iloc[
                                                                       -(n + 5):-n], how='outer')
                                    # continue
                                else:
                                    df_enter_angle = pd.merge(df_enter_angle,
                                                              df_collision_1[df_collision_1['ID'] == enter_id].tail(
                                                                  time_for_angle),
                                                              how='outer')

                                    df_enter_calculate_angle = pd.merge(df_enter_calculate_angle,
                                                                        df_collision_1[
                                                                            df_collision_1['ID'] == enter_id].iloc[
                                                                        -(2 * (n - 1) + time_for_angle):-2 * (n - 1)],
                                                                        how='outer')
                                    df_enter_calculate_real = pd.merge(df_enter_calculate_real,
                                                                       df_collision_1[
                                                                           df_collision_1['ID'] == enter_id].iloc[
                                                                       -(n + time_for_angle):-n], how='outer')

                                # 单独处理入轨迹的角速度
                            df_enter_angle['Coordinates'] = df_enter_angle.apply(lambda row: (row['X'], row['Y']),
                                                                                 axis=1)
                            #         for i in range(df_collision_enter.shape[0]):

                            df_enter_calculate_angle['Coordinates'] = df_enter_calculate_angle.apply(
                                lambda row: (row['X'], row['Y']),
                                axis=1)
                            df_enter_calculate_real['Coordinates'] = df_enter_calculate_real.apply(
                                lambda row: (row['X'], row['Y']),
                                axis=1)

                            # 重新设置索引，否则会根据原索引传输值，出现问题
                            df_enter_angle = df_enter_angle.reset_index(drop=True)
                            df_enter_calculate_angle = df_enter_calculate_angle.reset_index(drop=True)
                            df_enter_calculate_real = df_enter_calculate_real.reset_index(drop=True)

                            df_enter_angle['last_point_Coordinates'] = df_enter_calculate_angle['Coordinates']
                            df_enter_angle['last_point_time'] = df_enter_calculate_angle['Timestamp']
                            df_enter_angle['real_point_Coordinates'] = df_enter_calculate_real['Coordinates']

                            # df_enter_angle.to_csv(r'C:\Users\Windows11\Desktop\df_enter_angle.csv')
                            # 开始神圣的计算吧！
                            for i in df_enter_angle.index:
                                vector_1 = np.subtract(df_enter_angle.loc[i, 'real_point_Coordinates'],
                                                       df_enter_angle.loc[i, 'Coordinates'])
                                vector_2 = np.subtract(df_enter_angle.loc[i, 'last_point_Coordinates'],
                                                       df_enter_angle.loc[i, 'real_point_Coordinates'])
                                # if len(vector_1) or len(vector_2) != 2:
                                # print(vector_1)
                                # print(vector_2)
                                # print(df_enter_angle)
                                #             print(vector_1)
                                #             print(vector_2)
                                if (np.sqrt(np.dot(vector_1, vector_1)) * np.sqrt(np.dot(vector_2, vector_2))) == 0:
                                    df_enter_angle.loc[i, 'plastance'] = np.nan
                                else:
                                    df_enter_angle.loc[i, 'plastance'] = (np.arccos(np.dot(vector_1, vector_2) / (
                                            np.sqrt(np.dot(vector_1, vector_1)) * np.sqrt(
                                        np.dot(vector_2, vector_2))))) / (abs(
                                        df_enter_angle.loc[i, 'Timestamp'] - df_enter_angle.loc[i, 'last_point_time']))
                                #             if df_enter_angle.loc[i,'plastance'] is np.nan:
                                #                 df_enter_angle.loc[i,'plastance']=0

                            for group in all_pairs:
                                for exit_id in group[1]:
                                    if len(df_collision_1[df_collision_1['ID'] == exit_id]) < 40:
                                        n = 1
                                        df_exit_angle = pd.merge(df_exit_angle,
                                                                 df_collision_1[df_collision_1['ID'] == exit_id].head(
                                                                     5),
                                                                 how='outer')
                                        df_exit_calculate_angle = pd.merge(df_exit_calculate_angle,
                                                                           df_collision_1[
                                                                               df_collision_1['ID'] == exit_id].iloc[
                                                                           2 * (n - 1): 5 + 2 * (n - 1)], how='outer')
                                        df_exit_calculate_real = pd.merge(df_exit_calculate_real,
                                                                          df_collision_1[
                                                                              df_collision_1['ID'] == exit_id].iloc[
                                                                          n: 5 + n], how='outer')

                                        # continue
                                    else:
                                        df_exit_angle = pd.merge(df_exit_angle,
                                                                 df_collision_1[df_collision_1['ID'] == exit_id].head(
                                                                     time_for_angle),
                                                                 how='outer')
                                        df_exit_calculate_angle = pd.merge(df_exit_calculate_angle,
                                                                           df_collision_1[
                                                                               df_collision_1['ID'] == exit_id].iloc[
                                                                           2 * (n - 1):time_for_angle + 2 * (n - 1)],
                                                                           how='outer')
                                        df_exit_calculate_real = pd.merge(df_exit_calculate_real,
                                                                          df_collision_1[
                                                                              df_collision_1['ID'] == exit_id].iloc[
                                                                          n:time_for_angle + n], how='outer')
                            #             print(df_collision_1[df_collision_1['ID']==exit_id].iloc[n:n+time_before_collision])
                            #             print("oooo")
                            df_exit_angle['Coordinates'] = df_exit_angle.apply(lambda row: (row['X'], row['Y']), axis=1)
                            df_exit_calculate_angle['Coordinates'] = df_exit_calculate_angle.apply(
                                lambda row: (row['X'], row['Y']),
                                axis=1)
                            df_exit_calculate_real['Coordinates'] = df_exit_calculate_real.apply(
                                lambda row: (row['X'], row['Y']),
                                axis=1)

                            # 重新设置索引，否则会根据原索引传输值，出现问题
                            df_exit_angle = df_exit_angle.reset_index(drop=True)
                            df_exit_calculate_angle = df_exit_calculate_angle.reset_index(drop=True)
                            df_exit_calculate_real = df_exit_calculate_real.reset_index(drop=True)

                            df_exit_angle['last_point_Coordinates'] = df_exit_calculate_angle['Coordinates']
                            df_exit_angle['last_point_time'] = df_exit_calculate_angle['Timestamp']
                            df_exit_angle['real_point_Coordinates'] = df_exit_calculate_real['Coordinates']

                            for i in df_exit_angle.index:
                                vector_1 = np.subtract(df_exit_angle.loc[i, 'Coordinates'],
                                                       df_exit_angle.loc[i, 'real_point_Coordinates'])
                                vector_2 = np.subtract(df_exit_angle.loc[i, 'real_point_Coordinates'],
                                                       df_exit_angle.loc[i, 'last_point_Coordinates'])
                            if (np.sqrt(np.dot(vector_1, vector_1)) * np.sqrt(np.dot(vector_2, vector_2))) == 0:
                                df_enter_angle.loc[i, 'plastance'] = np.nan
                            else:
                                df_exit_angle.loc[i, 'plastance'] = (np.arccos(np.dot(vector_1, vector_2) / (
                                        np.sqrt(np.dot(vector_1, vector_1)) * np.sqrt(np.dot(vector_2, vector_2))))) / (
                                                                    abs(
                                                                        df_exit_angle.loc[i, 'Timestamp'] -
                                                                        df_exit_angle.loc[i, 'last_point_time']))

                            df_exit_angle = df_exit_angle.fillna(0.000001)
                            df_enter_angle = df_enter_angle.fillna(0.000001)

                            return df_enter_angle, df_exit_angle

                        def opposite_merge(df_enter_angle, df_exit_angle, time_for_opposite):

                            #     chase_number =
                            #     time_for_opposite = 10
                            stay_still = 3
                            df_enter_opposite = df_enter_angle[df_enter_angle['ID'] < 0]
                            df_exit_opposite = df_exit_angle[df_exit_angle['ID'] < 0]

                            # 每个id只取前面的一些记录作为判据
                            for group_id, group_data in df_enter_angle.groupby('ID'):
                                df_enter_opposite = pd.merge(df_enter_opposite,
                                                             df_enter_angle[df_enter_angle['ID'] == group_id].tail(
                                                                 time_for_opposite),
                                                             how='outer')

                            for group_id, group_data in df_exit_angle.groupby('ID'):
                                df_exit_opposite = pd.merge(df_exit_opposite,
                                                            df_exit_angle[df_exit_angle['ID'] == group_id].head(
                                                                time_for_opposite),
                                                            how='outer')

                                # 开始神圣的计算吧！
                            for i in df_enter_opposite.index:
                                vector_1 = np.subtract(df_enter_opposite.loc[i, 'real_point_Coordinates'],
                                                       df_enter_opposite.loc[i, 'Coordinates'])
                                vector_2 = np.subtract(df_enter_opposite.loc[i, 'last_point_Coordinates'],
                                                       df_enter_opposite.loc[i, 'real_point_Coordinates'])
                                vector_3 = vector_1 + vector_2

                                df_enter_opposite.loc[i, 'vector_3[0]'] = vector_3[0]
                                df_enter_opposite.loc[i, 'vector_3[1]'] = vector_3[1]

                            # df_enter_opposite['vector_3'] = df_enter_opposite.apply(lambda row: (row['vector_3[0]'], row['vector_3[1]']),axis=1)

                            for i in df_exit_opposite.index:
                                vector_1 = np.subtract(df_exit_opposite.loc[i, 'Coordinates'],
                                                       df_exit_opposite.loc[i, 'real_point_Coordinates'])
                                vector_2 = np.subtract(df_exit_opposite.loc[i, 'real_point_Coordinates'],
                                                       df_exit_opposite.loc[i, 'last_point_Coordinates'])
                                vector_3 = vector_1 + vector_2

                                df_exit_opposite.loc[i, 'vector_3[0]'] = vector_3[0]
                                df_exit_opposite.loc[i, 'vector_3[1]'] = vector_3[1]

                            # df_exit_opposite['vector_3'] = df_exit_opposite.apply(lambda row: (row['vector_3[0]'], row['vector_3[1]']),axis=1)
                            #     df_enter_opposite.to_csv(r'C:\Users\Admin\Desktop\enter_opposite_1202.csv')
                            #     df_exit_opposite.to_csv(r'C:\Users\Admin\Desktop\exit_opposite_1202.csv')

                            id_vector_enter = {}
                            for group_id, group_data in df_enter_opposite.groupby('ID'):
                                corrent_vector_3 = []
                                if abs(group_data['vector_3[0]'].sum()) < stay_still:
                                    corrent_vector_3.append(0)
                                else:
                                    corrent_vector_3.append(group_data['vector_3[0]'].sum())
                                if abs(group_data['vector_3[1]'].sum()) < stay_still:
                                    corrent_vector_3.append(0)
                                else:
                                    corrent_vector_3.append(group_data['vector_3[1]'].sum())
                                id_vector_enter[group_id] = corrent_vector_3
                                corrent_vector_3 = []
                            # 这一步我想把id与合成的向量进行绑定，方便后续运算
                            #     print(id_vector_enter)

                            id_vector_exit = {}

                            for group_id, group_data in df_exit_opposite.groupby('ID'):
                                corrent_vector_3 = []
                                if abs(group_data['vector_3[0]'].sum()) < stay_still:
                                    corrent_vector_3.append(0)
                                else:
                                    corrent_vector_3.append(group_data['vector_3[0]'].sum())
                                if abs(group_data['vector_3[1]'].sum()) < stay_still:
                                    corrent_vector_3.append(0)
                                    # 如果在某一方向上移动的向量方向角度小于某个值，认为它在这个方向上移动不显著，设为0
                                    # 小于stay_still的阈值，说明这个轨迹在该方向约等于不动，坐标变化设为0
                                else:
                                    corrent_vector_3.append(group_data['vector_3[1]'].sum())
                                id_vector_exit[group_id] = corrent_vector_3
                                corrent_vector_3 = []
                            #         df_enter_opposite.to_csv('C:/Users/Admin/Desktop/df_enter_opposite.csv')
                            #         df_exit_opposite.to_csv('C:/Users/Admin/Desktop/df_exit_opposite.csv')
                            opposite_judge = []
                            #     print(all_pairs)
                            # for item in all_pairs:

                            corrent_judge = []
                            corrent_judge.append(pair[0])
                            #     print(np.multiply(np.array(id_vector_enter[item[0][0]]),np.array(id_vector_enter[item[0][1]]))[1])
                            #         print(item[0])
                            judge = np.multiply(np.array(id_vector_enter[pair[0][0]]),
                                                np.array(id_vector_enter[pair[0][1]]))

                            #     [x for x in judge if x ==0]
                            if len([x for x in judge if x < 0]) >= 1 or (
                                    len([x for x in judge if x < 0]) == 1 and len([x for x in judge if x == 0]) == 1):
                                corrent_judge.append(1)

                            else:
                                corrent_judge.append(0)
                            opposite_judge.append(corrent_judge)
                            # 这是判断是否为相向，1说明是相向，0说明是同向
                            # print(all_pairs)
                            # print("通过入轨迹判断是否相向，1说明是相向，0说明是同向")
                            # print(opposite_judge)
                            merge_pairs = []
                            corrent_pair = []
                            corrent_judge = []
                            merge_judge = []
                            # this_pairs = [pair]
                            for item in opposite_judge:
                                if item[1] == 1:
                                    # 只处理相向的情况
                                    for st_id in item[0]:
                                        corrent_judge.append(st_id)
                                        for end_id in \
                                        this_pairs[this_pairs.index(list(x for x in this_pairs if x[0] == item[0])[0])][
                                            1]:
                                            judge = np.multiply(np.array(id_vector_enter[st_id]),
                                                                np.array(id_vector_exit[end_id]))
                                            if len([x for x in judge if x > 0]) == 2:
                                                corrent_judge.append(1)
                                                # 这是单个入轨迹与两个出轨迹分别配对并赋分，出现上述情况，证明匹配，打高分
                                            else:
                                                if len([x for x in judge if x < 0]) == 2 or (
                                                        len([x for x in judge if x < 0]) == 1 and len(
                                                    [x for x in judge if x == 0]) == 1):
                                                    corrent_judge.append(-1)
                                                # 如果出现上述的情况，证明它们相当不匹配，给打低分
                                                else:
                                                    corrent_judge.append(0)
                                                    # 不符合上述情况，模棱两可，打零分
                                        merge_judge.append(corrent_judge)
                                        corrent_judge = []
                            # print("轨迹判断得分")
                            # print(merge_judge)
                            for item in merge_judge:
                                #     print(list(x for x in all_pairs if item[0] in x[0])[0][1][item.index(max(item[1],item[2]))-1])
                                #     print(all_pairs[all_pairs.index(list(x for x in all_pairs if item[0] in x[0][0])[1])
                                corrent_pair.append(item[0])
                                corrent_pair.append(
                                    list(x for x in this_pairs if item[0] in x[0])[0][1][
                                        item.index(max(item[1], item[2])) - 1])
                                # item[0]是目前在研究的入轨迹，后面两个是两个出轨迹，分别与它进行匹配，取匹配分数最高的那个，认为它与该入轨迹匹配
                                merge_pairs.append(corrent_pair)

                                corrent_pair = []
                            # print("轨迹配对选择")
                            # print(merge_pairs)
                            # merge_pairs.append('R to R')
                            return merge_pairs

                        this_pairs = [pair]
                        all_id = []
                        for item in this_pairs:
                            considered_id = [t for t in item]
                            for issue in considered_id:
                                for t in issue:
                                    all_id.append(t)
                        df_collision_1 = track_df[track_df['ID'].isin(all_id)]
                        # df_collision_1.to_csv('C:/Users/Admin/Desktop/df_collision_1.csv')
                        time_before_collision = 100  # 用于速度
                        # if min(df_collision_1.loc[:,'ID'].value_counts()) < 200:
                        #     # print(min(df_collision_1.loc[:,'ID'].value_counts()))
                        #     time_for_angle = int(min(df_collision_1.loc[:,'ID'].value_counts())/2)  # 用于角速度
                        #     # time_for_opposite = min(df_collision_1.loc[:,'ID'].value_counts())
                        #     time_for_opposite = 20
                        # else:
                        time_for_angle = 20  # 用于角速度
                        time_for_opposite = 5
                        # df_collision

                        df_collision_enter = df_collision_exit = df_enter_calculate = df_exit_calculate = \
                        df_collision_1[
                            df_collision_1['ID'] < 0]
                        df_enter_angle = df_exit_angle = df_enter_calculate_angle = df_exit_calculate_angle = df_enter_calculate_real = df_exit_calculate_real = \
                            df_collision_1[df_collision_1['ID'] < 0]
                        # print(df_enter_angle)

                        # df_enter_angle, df_exit_angle = angle_calculate(pair,5)
                        # df_enter_v, df_exit_v = velocity_calculate(pair,10)
                        df_enter_angle_1, df_exit_angle_1 = angle_calculate(df_enter_angle, df_exit_angle,
                                                                            df_enter_calculate_angle,
                                                                            df_exit_calculate_angle,
                                                                            df_enter_calculate_real,
                                                                            df_exit_calculate_real, pair, 2)

                        merged_pair = opposite_merge(df_enter_angle_1, df_exit_angle_1, time_for_opposite)
                        return merged_pair

                    def collision_type_recognize(time_window, track_df, all_pairs, mean_threshold, err_threshold):
                        paired_list = []
                        # 判断类型待选：R&R:1, R&D:2, D&D:3
                        # time_window = 50
                        for pair in all_pairs:
                            df_enter = pd.concat([track_df[track_df['ID'] == pair[0][0]].tail(time_window),
                                                  track_df[track_df['ID'] == pair[0][1]].tail(time_window)])
                            df_exit = pd.concat([track_df[track_df['ID'] == pair[1][0]].head(time_window),
                                                 track_df[track_df['ID'] == pair[1][1]].head(time_window)])
                            # 提取大时间窗内的数据

                            df_enter['Coordinates'] = df_enter.apply(lambda row: (row['X'], row['Y']), axis=1)
                            df_exit['Coordinates'] = df_exit.apply(lambda row: (row['X'], row['Y']), axis=1)
                            id_center = {}
                            for group_id, group_data in df_enter.groupby('ID'):
                                x = group_data['X']
                                y = group_data['Y']
                                points = list(zip(x, y))
                                points_np = np.array(points)
                                rect = cv2.minAreaRect(points_np)
                                center, size, angle = rect
                                width, height = size
                                # print(center)

                                for i in group_data.index:
                                    df_enter.loc[i, 'dist_to_center'] = math.dist(df_enter.loc[i, 'Coordinates'],
                                                                                  center)
                                    id_center[group_id] = center
                            df_enter['center'] = df_enter['ID'].map(id_center)
                            # df_enter.loc[df_enter['ID']==group_id,'center'] = [center[0], center[1]]
                            # print(df_enter)
                            id_center = {}
                            for group_id, group_data in df_exit.groupby('ID'):
                                x = group_data['X']
                                y = group_data['Y']
                                points = list(zip(x, y))
                                points_np = np.array(points)
                                rect = cv2.minAreaRect(points_np)
                                center, size, angle = rect
                                width, height = size
                                for i in group_data.index:
                                    df_exit.loc[i, 'dist_to_center'] = distance.euclidean(df_exit.loc[i, 'Coordinates'],
                                                                                          center)

                                    id_center[group_id] = center
                                df_exit['center'] = df_exit['ID'].map(id_center)

                            # mean_threshold = 15
                            # err_threshold = 0.5

                            # 计算与轨迹中心的距离平均值&标准差

                            pair_judge_0 = {}  # 用来装该碰撞对的判据0
                            corrent_pair_data = {}
                            # 这个字典存储每一个pair的入轨迹、出轨迹的距离平均值和std，作为坐标散度度量
                            # corrent_pair_data = {start_id_1:[mean,std],start_id_2:[mean,std],end_id_1:[mean,std],end_id_2:[mean,std]}
                            for start_id in pair[0]:
                                corrent_id_data = []  # 用来存储该id轨迹的mean和std
                                mean = df_enter[df_enter['ID'] == start_id]['dist_to_center'].mean()
                                err = df_enter[df_enter['ID'] == start_id]['dist_to_center'].std() / np.sqrt(
                                    len(df_enter[df_enter['ID'] == start_id]['dist_to_center']))
                                corrent_id_data.append(mean)

                                corrent_id_data.append(err)
                                corrent_pair_data[start_id] = corrent_id_data

                                if mean > mean_threshold and err > err_threshold:
                                    pair_judge_0[start_id] = 1
                                else:
                                    pair_judge_0[start_id] = 0

                            for end_id in pair[1]:
                                corrent_id_data = []  # 用来存储该id轨迹的mean和std
                                mean = df_exit[df_exit['ID'] == end_id]['dist_to_center'].mean()
                                err = df_exit[df_exit['ID'] == end_id]['dist_to_center'].std() / np.sqrt(
                                    len(df_exit[df_exit['ID'] == end_id]['dist_to_center']))
                                corrent_id_data.append(mean)
                                corrent_id_data.append(err)
                                corrent_pair_data[end_id] = corrent_id_data

                                if mean > mean_threshold and err > err_threshold:
                                    pair_judge_0[end_id] = 1
                                else:
                                    pair_judge_0[end_id] = 0
                            # 若一个轨迹的元素 1 及 元素 2 均大于各自给定的阈值，可以认为是roaming状态，否则是dwelling，判据 0
                            # 轨迹是roaming：判据 0 = 1
                            # 轨迹是dwelling：判据0 = 0

                            # 判断层：
                            # 若入、出轨迹均是一R &一D（判据 0 为1 ：1）：进入case 1
                            if (list(pair_judge_0.values())[0] + list(pair_judge_0.values())[1]) == 1 & (
                                    list(pair_judge_0.values())[2] + list(pair_judge_0.values())[3]) == 1:
                                merge_pair = case_1(pair, pair_judge_0)
                                why = 1
                                paired_list.append(merge_pair)
                                continue
                            else:
                                # 若入、出轨迹均是 D（判据 0 为0 ：0）：进入case 2
                                if (list(pair_judge_0.values())[0] + list(pair_judge_0.values())[1]) == 0 & (
                                        list(pair_judge_0.values())[2] + list(pair_judge_0.values())[3]) == 0:
                                    merge_pair = case_2(df_enter, df_exit, pair)
                                    why = 2
                                    paired_list.append(merge_pair)
                                    continue
                                else:
                                    # 若入、出轨迹均是 R（判据 0 为2 ：2）：进入case 3
                                    if (list(pair_judge_0.values())[0] + list(pair_judge_0.values())[1]) == 2 & (
                                            list(pair_judge_0.values())[2] + list(pair_judge_0.values())[3]) == 2:
                                        merge_pair = case_3(pair, track_df)
                                        why = 3
                                        paired_list.append(merge_pair)
                                        continue
                                    else:
                                        merge_pair = case_3(pair, track_df)
                                        # paired_list.append(merge_pair)
                                        # merge_pair = ['无法识别']
                                        paired_list.append(merge_pair)
                                        continue
                                # else:
                                #     print('无法识别')

                            # paired_list.append(merge_pair)
                        for pair in paired_list:
                            if (len(pair) != 0 and pair[0][1] == pair[1][1]):
                                paired_list.remove(pair)

                        paired_list = [pair for pair in paired_list if pair != []]

                        return paired_list

                    merge_list = collision_type_recognize(100, df_after_second_clear, all_pairs, 20, 0.5)

                    test_list = merge_list
                    drop_list = []  # 这是删除某一个中间的小配对用的
                    all_list = []  # 这是首尾相连后得到的所有配对
                    corrent_list = copy.deepcopy(test_list)  # 这是删除当前小配对后剩下的内容

                    last_elements = []
                    for item in test_list:
                        if len(item) != 0:
                            last_elements.append(item[-1])
                    for item in test_list:
                        if len(item) == 0:
                            continue
                        else:
                            if ((item in drop_list) == False) and last_elements.count(item[-1]) == 1:
                                corrent_list.remove(item)
                                #         print(corrent_list)
                                for issue in corrent_list:
                                    if issue[0] == item[-1]:
                                        item = list(set(item).union(set(issue)))
                                        item.sort()
                                        # corrent_list.remove(issue)
                                        drop_list.append(issue)
                                        continue
                                    else:
                                        continue
                                all_list.append(item)
                            corrent_list = copy.deepcopy(test_list)

                        for item in all_list:
                            for i in item:
                                df_after_second_clear.loc[df_after_second_clear['ID'].isin(i), 'ID'] = min(i)

        ##############################################################
        # 将因为经过噪音而没法merge的轨迹连接起来
        df_merge_update = df_after_second_clear.copy()

        # df_merge_update=df_after_second_clear.copy()
        diagonal_lengths_rate_update = {}
        diagonal_lengths_update = {}
        Misalignment_area_update = {}
        Misalignment_rate_update = {}
        rect_center_update = {}
        rect_area_update = {}
        plt.figure()
        for group_id, group_data in df_merge_update.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            time_1 = group_data['Timestamp']
            start_time = time_1.iloc[0]
            end_time = time_1.iloc[-1]
            #     plt.text(x.iloc[0],y.iloc[0],str(group_id)+":s"+str(start_time))
            #     plt.text(x.iloc[-1],y.iloc[-1],str(group_id)+":e"+str(end_time))
            plt.title('tracks after merge-ID num:' + str(len(df_merge_update['ID'].unique())))
            plt.plot(x, y)

            points = list(zip(x, y))
            points_np = np.array(points)
            rect = cv2.minAreaRect(points_np)
            center, size, angle = rect
            width, height = size
            time_diff = time_1.iloc[-1] - time_1.iloc[0]
            Misalignment_rate_update[group_id] = round((len(set(list(map(itemgetter(0), points))))) / time_diff, 2)
            Misalignment_area_update[group_id] = len(set(list(map(itemgetter(0), points))))
            diagonal_lengths_update[group_id] = round((np.sqrt(width ** 2 + height ** 2)), 2)
            diagonal_lengths_rate_update[group_id] = round((np.sqrt(width ** 2 + height ** 2)) / time_diff, 2)
            rect_center_update[group_id] = center
            rect_area_update[group_id] = width * height
        df_merge_update['Misalignment_area'] = df_merge_update['ID'].map(Misalignment_area_update)
        df_merge_update['Misalignment_rate'] = df_merge_update['ID'].map(Misalignment_rate_update)
        df_merge_update['Diagonal_Length'] = df_merge_update['ID'].map(diagonal_lengths_update)
        df_merge_update['Diagonal_Length_rate'] = df_merge_update['ID'].map(diagonal_lengths_rate_update)
        df_merge_update['rect_center'] = df_merge_update['ID'].map(rect_center_update)
        df_merge_update['rect_area'] = df_merge_update['ID'].map(rect_area_update)
        # ppath+ '/' +str(file_name[:-4]) +'_raw_data.csv')

        df_after_second_clear = df_merge_update.copy()
        plt.figure()
        for group_id, group_data in df_after_second_clear.groupby('ID'):
            x = group_data['X']
            y = group_data['Y']
            time_2 = group_data['Timestamp']
            start_time = time_2.iloc[0]
            end_time = time_2.iloc[-1]
            plt.plot(x, y)

        plt.title('after noise-cleared twice-ID num:' + str(len(df_after_second_clear['ID'].unique())))
        plt.savefig(save_path + '/' + name_list[3] + '.png', bbox_inches='tight')
        if show:
            plt.show()
        id_list.append(len(df_after_second_clear['ID'].unique()))
        df_after_second_clear.to_csv(save_path + '/' + str(file_name[:-4]) + '_second_clear.csv')

        color_list = ['#F08080', '#FFDEAD', '#20B2AA', '#ADD8E6']

        plt.figure(figsize=(15, 10))
        plt.bar(range(len(id_list)), id_list, color='#f9766e', tick_label=name_list[0:4])
        plt.bar(range(len(id_list)), id_list, color=color_list, tick_label=name_list[0:4])
        plt.title('Compare the ID number of each process')
        # print(len(id_list))
        # print("aaa")
        for i in range(len(id_list)):
            # print(number)

            plt.text(i, id_list[i] + 0.1, id_list[i], fontsize=16, ha='center', va='bottom')
        plt.savefig(save_path + '/' + 'bar.png')
        if show:
            plt.show()
        plt.close('all')
        merge_num += 1
        print('已完成' + str(merge_num) + '/' + str(len(data_files)))
    print('已完成merge，祝数据分析顺利')
