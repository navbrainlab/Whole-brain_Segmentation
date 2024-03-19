import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gjh
import copy


def multi_processing_draw(tmp, version=f'viz_result_e6d6_max_pm_exp_nerve_v16'):
    
    fig, axes = plt.subplots(figsize=(5,10))
    temp_4_align = np.load('Data/test_tracking/real_jeff_all/real_808.npy')
    
    # tmp = match_clip['temp_vol_name']


    temp_path = f'Data/test_tracking/real_jeff_all/{tmp}'
    n_points = np.load(temp_path)
    n_points = n_points[:, :3]
    num1=27
    num2=10
    print(n_points[num1,:],n_points[num2,:])
    print(distance_3d_numpy(n_points[num1,:],n_points[num2,:]))

    #------------get aligned temp and test, for draw compare
    n_points = gjh.get_angle_aligned_points(n_points, temp_4_align)
    
    # print(n_points[87,:],n_points[112,:])


    # 绘制散点图
    t_temp_pts = copy.deepcopy(n_points) + np.array([0,0,0])


    gjh.limt_xy(axes, x_range=(-0.5, 0.5), y_range=(-0.75, 1.2))
    im = axes.scatter(t_temp_pts[:, 0], t_temp_pts[:, 1], c='gray', s=2)  # draw test,test整体坐标往右移动2

    # title = str(gt_of_match_false_test_id_str[i])

    #111111111111, 画出test中预测错误的index实际在temp中应该是哪个id---------------------------------
    # axes.text(-0.25, 1, f'test vol, pred {title} as ({pred_false_as_which_temp_id_str[i]})', fontsize=5, color='red')
    gjh.draw_pts_with_color(axes, t_temp_pts, np.arange(t_temp_pts.shape[0]),
                    color_intensity=np.ones(t_temp_pts.shape[0]),
                    txt= [str(i) for i in (range(len(t_temp_pts)))],
                    color='red')
    plt.title(tmp)
    plt.savefig('/mnt/c/gaochao/CODE/NeuronTrack/fDNC_Neuron_ID/analyze_results/check distributaion/a.png',dpi=800)
   

def distance_3d_numpy(point1, point2):
    distance = np.linalg.norm(point2 - point1)
    return distance

if __name__ == '__main__':
    print('here')
    multi_processing_draw('real_77.npy')
