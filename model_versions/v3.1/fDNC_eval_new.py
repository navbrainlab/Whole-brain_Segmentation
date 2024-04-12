from scipy.optimize import linear_sum_assignment
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import sys
sys.path.append('/mnt/c/gaochao/CODE/NeuronTrack/fDNC_Neuron_ID/model_versions/3')
from model_new import NIT_Registration, neuron_data_pytorch
sys.path.append('/mnt/c/gaochao/CODE/NeuronTrack/fDNC_Neuron_ID/src')
from cpd_rigid_sep import register_rigid
from cpd_nonrigid_sep import register_nonrigid
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from tqdm import tqdm




def find_match_dis(mov, ref, match_dict):
    dis_m = np.sum((mov[:, np.newaxis, :] - ref[np.newaxis, :, :]) ** 2, axis=2)
    dis_list = dis_m[match_dict[:, 0], match_dict[:, 1]]
    return dis_list


def cpd_match(mov, ref, match_dict, method='max', plot=False):
    w = 0.1
    lamb = 4e3
    #lamb = 2e3
    beta = 0.25
    # cpd transform
    mov_rigid, _, _, sigma2 = register_rigid(ref, mov, w=w, fix_scale=True)
    ori_dis = find_match_dis(mov_rigid, ref, match_dict)
    mov_rigid_inv = np.copy(mov_rigid)
    mov_rigid_inv[:, :2] *= -1
    mov_rigid_inv, _, _, sigma2 = register_rigid(ref, mov_rigid_inv, w=w, fix_scale=True)

    inv_dis = find_match_dis(mov_rigid_inv, ref, match_dict)
    if np.mean(ori_dis) > np.mean(inv_dis):
        mov_new = mov_rigid_inv
    else:
        mov_new = mov_rigid

    mov_nonrigid = register_nonrigid(ref, mov_new, w=w, lamb=lamb, beta=beta)
    # plot the results.
    if plot:
        plt.scatter(mov[:, 0], mov[:, 1], c='red')
        plt.scatter(mov_new[:, 0], mov_new[:, 1], c='yellow')
        plt.scatter(mov_nonrigid[:, 0], mov_nonrigid[:, 1], c='green')
        plt.scatter(ref[:, 0], ref[:, 1], c='black')
        plt.show()

    dis_m = np.sum((mov_nonrigid[:, np.newaxis, :] - ref[np.newaxis, :, :]) ** 2, axis=2)
    if method == 'max':
        col = np.argmin(dis_m, axis=1)
        row = np.array(range(dis_m.shape[0]))
    else:
        row, col = linear_sum_assignment(dis_m)

    return row, col


def jeff_match(pSNew, idx1, idx2, track_method=0):
    # use pointStatsnew label.
    names = pSNew.dtype.names
    pS_dictName = dict()
    for i, name in enumerate(names): pS_dictName[name] = i
    if track_method == 0:
        trackIdx = pSNew[0, idx1][pS_dictName['trackIdx']]
    else:
        trackIdx = pSNew[0, idx1][pS_dictName['trackIdx_1']]
    num_raw = pSNew[0, idx1][pS_dictName['rawPoints']].shape[0]
    trackIdx1 = trackIdx[:num_raw, 0]
    if track_method == 0:
        trackIdx = pSNew[0, idx2][pS_dictName['trackIdx']]
    else:
        trackIdx = pSNew[0, idx2][pS_dictName['trackIdx_1']]

    num_raw = pSNew[0, idx2][pS_dictName['rawPoints']].shape[0]
    trackIdx2 = trackIdx[:num_raw, 0]

    track_dict = dict()
    for i, trackid in enumerate(trackIdx2):
        if not np.isnan(trackid) and trackid != 0:
            track_dict[trackid] = i
    row = []
    col = []
    for i, trackid in enumerate(trackIdx1):
        if not np.isnan(trackid) and trackid != 0 and trackid in track_dict:
            row.append(i)
            col.append(int(track_dict[trackid]))
    return np.array(row), np.array(col)

# python fDNC_eval.py --eval_path ../Data/test_neuropal_Chaudhary

# python fDNC_eval.py --eval_path ../Data/test_neuropal_Chaudhary --n_hidden 48 --n_layer 8 --model_path ../model/nitReg_nh48_nl8_ft1_dataall_elam_0.1_627.bin

# python fDNC_eval.py --eval_path ../Data/test_synthetic --n_hidden 128 --n_layer 6 --model_path ../model/nitReg_nh128_nl6_ft0_dataall_elam_0.1_627.bin

# python fDNC_eval.py --eval_path ../Data/test_synthetic --n_hidden 128 --n_layer 6 --model_path ../model/nitReg_nh128_nl6_ft0_dataall_elam_0.1_628.bin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path",
                        default="../pretrained_model/model.bin",
                        type=str)

    parser.add_argument("--eval_path", default="../Data/test_synthetic", type=str) # the path to test data.
    parser.add_argument("--save", default=0, type=int)
    parser.add_argument("--save_p", default="../results", type=str)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--n_hidden", default=128, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--method", default='max', type=str)
    parser.add_argument("--show_name", default=0, type=int)
    parser.add_argument("--ref_idx", default=5, type=int) #5 is used as template for neuropal data.
    parser.add_argument("--do_cpd", default=0, type=int)
    parser.add_argument("--do_jeff", default=0, type=int)
    parser.add_argument("--cal_lim", default=200000, type=int)
    parser.add_argument("--acc_name", default='xinwei', type=str)
    parser.add_argument("--conf_thd", default=0.00, type=float)
    parser.add_argument("--tmp_p", default="no", type=str)
    parser.add_argument("--n_outlier", default=5, type=int)

    parser.add_argument("--flag", default='a', type=str)

    args = parser.parse_args()

    if args.tmp_p == 'no':
        tmp_p = None
    else:
        tmp_p = args.tmp_p


    if args.do_jeff:
        psNew_p = '../Data/NERVE/PointsStatsNew2.mat'
        pSNew = sio.loadmat(psNew_p)['pointStatsNew']

    dev_data = neuron_data_pytorch(args.eval_path, batch_sz=args.batch_size, shuffle=False, mode='all',
                                   ref_idx=args.ref_idx, show_name=True, shuffle_pt=False, tmp_path=tmp_p)
    dev_data_loader = DataLoader(dev_data, shuffle=False, num_workers=1, collate_fn=dev_data.custom_collate_fn,drop_last=False)
    if args.show_name:
        for i, b in enumerate(dev_data.bundle_list):
            print("{}:{}")

    # load model
    model = NIT_Registration(input_dim=3, n_hidden=args.n_hidden, cuda=args.cuda, n_layer=args.n_layer, p_rotate=0, feat_trans=0)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    params = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    model = model.to(device)

    num_hit_all = num_hit_cpd_all = num_hit_jeff_all = num_hit_jeff_all_1 = 0
    num_match_all = num_match_cpd_all = num_match_jeff_all = num_match_jeff_all_1 = 0

    # for calculating upper and lower bound for Hang's data
    num_hit_upper = num_match_label_all = 0

    num_hit_list = [0] * 100
    num_pair = 0
    save_idx = 0

    num_cal = 0

    tfmer_list = list()
    cpd_list = list()
    jeff_list = list()
    jeff_list_1 = list()

    thd_ratio_list = list()

    time_trans_list = list()
    time_hung_list = list()
    time_cpd_list = list()
    num_match_list = list()
    num_label_list = list()

    conf_dict = dict() # this dictionary store info about confidence and accuracy
    conf_dict['conf_result'] = [] # list for (confidence, correct) pairs.
    num_seg = []

    num_pm = 25
    score_list = [[] for i in range(num_pm)]
    num_hit_all = [0]*num_pm
    num_match_all = [0]*num_pm
    num_match_label_all = [0]*num_pm
    num_hit_upper = [0]*num_pm
    score_list_top_n = [[] for i in range(num_pm)] #-----------------------
    all_batch_name2embedd = [{} for i in range(num_pm)]

    match_info = [[] for i in range(num_pm)]
    for batch_idx, data_batch in enumerate(dev_data_loader):
        # if num_cal > args.cal_lim:
        #     break
        tic = time.time()
        model.eval()
        print('batch:{}'.format(batch_idx))

        pt_batch = data_batch['pt_batch']
        match_dict = data_batch['match_dict']
        label = data_batch['pt_label']


        
        from src_utils import *
        new_pt_batch = []
        for pts in pt_batch:
            pts = get_angle_aligned_points(pts,pt_batch[data_batch['ref_i']])
            new_pt_batch.append(pts)
        pt_batch = new_pt_batch

        #print(pt_batch)
    #for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
        #loss = -model(src_sents, tgt_sents).sum()

        with torch.no_grad():
            _, output_pairs = model(pt_batch, match_dict=match_dict, ref_idx=data_batch['ref_i'], mode='eval')
        num_worm = output_pairs['p_m'][0].size(0)
        time_trans_list.append(time.time() - tic)
        # p_m is the match of worms to the worm0
        # breakpoint()
        for pm_idx,output_pairs_p_m in enumerate(output_pairs['p_m']):
            num_seg = []
            thd_ratio_list = list()

            time_trans_list = list()
            time_hung_list = list()
            time_cpd_list = list()
            num_match_list = list()
            num_label_list = list()


            conf_dict = dict() # this dictionary store info about confidence and accuracy
            conf_dict['conf_result'] = [] # list for (confidence, correct) pairs.
            num_seg = []
            for i in range(0, num_worm):
                num_neui = len(pt_batch[i])
                num_seg.append(num_neui)
                num_label_cur = np.sum(label[i] >= 0)
                num_label_list.append(num_label_cur)

                if i == data_batch['ref_i']:
                    continue
                num_cal += 1
                tic = time.time()
                p_m = output_pairs_p_m[i].detach().cpu().numpy()

                p_m = p_m[:num_neui, :]
                if args.method == 'hung':
                    p_m = np.hstack((p_m[:, :-1], np.repeat(p_m[:, -1:], args.n_outlier,
                                                            axis=1)))
                    row, col = linear_sum_assignment(-p_m)
                if args.method == 'max':
                    col = np.argmax(p_m, axis=1)
                    row = np.array(range(num_neui))
                    cols = np.argsort(-p_m, axis=1)#--------------------


                num_row_ori = len(row)
                if args.conf_thd > 0:
                    row_mask = p_m[row, col] >= np.log(args.conf_thd)
                else:
                    row_mask = p_m[row, col] >= -float('Inf')
                thd_ratio = np.sum(row_mask) / max(1, len(row))
                print('thd_ratio:{}'.format(thd_ratio))
                thd_ratio_list.append(thd_ratio)
                row = row[row_mask]
                col = col[row_mask]
                cols = cols[row_mask]#---------------------------

                time_hung_list.append(time.time()-tic)
                if args.save:
                    save_file = os.path.join(args.save_p, 'match_{}.pt'.format(save_idx))
                    out_dict = dict()
                    out_dict['ref_pt'] = pt_batch[data_batch['ref_i']]
                    out_dict['ref_label'] = label[data_batch['ref_i']]
                    out_dict['mov_pt'] = pt_batch[i]
                    out_dict['ref_name'] = data_batch['pt_names'][data_batch['ref_i']]
                    out_dict['mov_name'] = data_batch['pt_names'][i]
                    cur_label = np.concatenate((label[data_batch['ref_i']], np.array([-2])))

                    out_dict['mov_label'] = cur_label[col]
                    out_dict['gt_label'] = np.ones(pt_batch[i].shape[0]) * -1
                    out_dict['gt_label'][match_dict[i][:, 0]] = label[data_batch['ref_i']][match_dict[i][:, 1]]
                    out_dict['origin_label'] = label[i]
                    out_dict['col'] = col



                log_p = np.mean(p_m[row, col])
                print('temp:{}, mov:{}'.format(data_batch['pt_names'][data_batch['ref_i']], data_batch['pt_names'][i]))
                jeff_idx2 = int(data_batch['pt_names'][data_batch['ref_i']].split('.npy')[0].split('_')[-1])
                jeff_idx1 = int(data_batch['pt_names'][i].split('.npy')[0].split('_')[-1])
                print(jeff_idx1, jeff_idx2)
                print('avg log prob:{}'.format(log_p))

                gt_match = match_dict[i]
                gt_match_dict = dict()
                for gt_m in gt_match:
                    gt_match_dict[gt_m[0]] = gt_m[1]

                num_match = num_hit = 0


                # get the list for confidence and correct or not.
                # ---------------------------------------------------------------------------origin-------------------------------------------------------------------------------
                # cur_conf_corr = []
                # for r_idx, r in enumerate(row):
                #     if r in gt_match_dict:
                #         num_match += 1
                #         if gt_match_dict[r] == col[r_idx]:
                #             num_hit += 1
                #             cur_conf_corr.append([p_m[r, col[r_idx]], True])
                #         else:
                #             cur_conf_corr.append([p_m[r, col[r_idx]], False])

                # conf_dict['conf_result'].append(cur_conf_corr)
                # ---------------------------------------------------------------------------new for save-------------------------------------------------------------------------------
                temp_vol_name = data_batch['pt_names'][data_batch['ref_i']].split('/')[-1]
                test_vol_name = data_batch['pt_names'][i].split('/')[-1]
                cur_conf_corr = []
                cur_match_info = {'match_right':[],'match_false':[],'p_m_exp':p_m }
                for r_idx, r in enumerate(row):
                    if r in gt_match_dict:
                        num_match += 1
                        if gt_match_dict[r] == col[r_idx]:
                            num_hit += 1
                            cur_conf_corr.append([p_m[r, col[r_idx]], True])
                            cur_match_info['match_right'].append([r, col[r_idx], gt_match_dict[r], p_m[r, col[r_idx]]])
                        else:
                            cur_conf_corr.append([p_m[r, col[r_idx]], False])
                            cur_match_info['match_false'].append([r, col[r_idx], gt_match_dict[r], p_m[r, col[r_idx]]])

                conf_dict['conf_result'].append(cur_conf_corr)
                # ---------------------------------------------------------------------------new for save-------------------------------------------------------------------------------

                # breakpoint()
                num_match = len(gt_match_dict)
                acc_m = num_hit / max(num_match, 1)
                num_match_list.append(max(num_match, 1))
                num_hit_all[pm_idx] += num_hit
                num_match_all[pm_idx] += num_match

                # calculate all the labelled neurons and upper/lower bound
                num_label_cur = np.sum(label[i] >= 0)

                print('num match:{}, num_label:{}'.format(num_match, num_label_cur))
                num_hit_upper[pm_idx] += num_hit + num_label_cur - num_match
                num_match_label_all[pm_idx] += num_label_cur

                print('num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
                # ---------------------------------------------------------------------------new for save2-------------------------------------------------------------------------------
                temp_dict = {'num match':num_match, 'num_label':num_label_cur,"num_hit":num_hit,"accuracy":acc_m,'test_vol_name':test_vol_name,'temp_vol_name':temp_vol_name}
                cur_match_info = {**cur_match_info,**temp_dict}
                match_info[pm_idx].append(cur_match_info)
                # ---------------------------------------------------------------------------new for save2-------------------------------------------------------------------------------
                num_hit_t_n = np.zeros(10) #---------------------
                for r_idx, r in enumerate(row):
                    if r in gt_match_dict:
                        for n in range(10):
                            if gt_match_dict[r] in cols[r_idx][:n+1]:
                                num_hit_t_n[n] += 1
                score_list_top_n[pm_idx].append(num_hit_t_n / max(num_match,1))#---------------------

                # breakpoint()
                score_list[pm_idx].append(acc_m)
                tfmer_list.append(acc_m)
                if args.save:
                    out_dict['accuracy'] = acc_m
                    with open(save_file, 'wb') as f:
                        pickle.dump(out_dict, f)
                        f.close()
                    save_idx += 1

                # get the top rank for gt match.
                num_pair += len(gt_match)
                for gt_m in gt_match:
                    topn = np.sum(p_m[gt_m[0]] >= p_m[gt_m[0], gt_m[1]])
                    for i_rank in range(10):
                        if topn <= i_rank+1:
                            num_hit_list[i_rank] += 1
            if pm_idx == 5:
                try: #--------------------------------------------------------------------------------------------------------------------
                    test_names = [i.split('/')[-1] for i in data_batch['pt_names']]
                    test_embedds = output_pairs['save_embedd'][pm_idx]['test'].detach().cpu()
                    test_name2embedd = {n:e for n,e in zip(test_names, test_embedds)}
                    test_name2embedd[temp_vol_name] = output_pairs['save_embedd'][pm_idx]['temp']
                    all_batch_name2embedd[pm_idx] = {**all_batch_name2embedd[pm_idx],**test_name2embedd}
                except:
                    pass
                #----------------------------------------------------------------------------------------------------------------------
            if args.do_cpd:
            # performance of cpd
                tic = time.time()
                num_match = num_hit = 0
                row_cpd, col_cpd = cpd_match(pt_batch[i], pt_batch[data_batch['ref_i']], gt_match, method=args.method, plot=False)

                for r_idx, r in enumerate(row_cpd):
                    if r in gt_match_dict:
                        num_match += 1
                        if gt_match_dict[r] == col_cpd[r_idx]:
                            num_hit += 1
                num_match = len(gt_match_dict)
                acc_m = num_hit / max(num_match, 1)
                num_hit_cpd_all += num_hit
                num_match_cpd_all += num_match
                print('CPD, num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
                cpd_list.append(acc_m)
                time_cpd_list.append(time.time()-tic)

            if args.do_jeff:
                # see the performance of Jeff's code
                row_jeff, col_jeff = jeff_match(pSNew, jeff_idx1, jeff_idx2)
                num_match = num_hit = 0
                for r_idx, r in enumerate(row_jeff):
                    if r in gt_match_dict:
                        num_match += 1
                        if gt_match_dict[r] == col_jeff[r_idx]:
                            num_hit += 1
                num_match = len(gt_match_dict)
                acc_m = num_hit / max(num_match, 1)
                num_hit_jeff_all += num_hit
                num_match_jeff_all += num_match
                print('Jeff, num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
                jeff_list.append(acc_m)

                row_jeff, col_jeff = jeff_match(pSNew, jeff_idx1, jeff_idx2, track_method=1)
                num_match = num_hit = 0
                for r_idx, r in enumerate(row_jeff):
                    if r in gt_match_dict:
                        num_match += 1
                        if gt_match_dict[r] == col_jeff[r_idx]:
                            num_hit += 1
                num_match = len(gt_match_dict)
                acc_m = num_hit / max(num_match, 1)
                num_hit_jeff_all_1 += num_hit
                num_match_jeff_all_1 += num_match
                print('Jeff_1, num_hit:{}, num_match:{}, accuracy:{}'.format(num_hit, num_match, acc_m))
                jeff_list_1.append(acc_m)


    a = [np.stack(i) for i in score_list_top_n[:6]]
    a = [np.mean(i,axis=0) for i in a][-1]
    print(a)
    prefix = args.flag
    np.save(f'analyze_results/top_n_acc/{prefix}_top_n_acc.npy',a)
    np.save(f'analyze_results/{prefix}_match_info.npy',np.array(match_info,dtype=object))
    np.save(f'analyze_results/save_embedd/{prefix}_embedd.npy',np.array(all_batch_name2embedd,dtype=object))

    # print('accuracy:{}'.format(num_hit_all / max(1, num_match_all)))
    # num_hit_list = np.array(num_hit_list) / num_pair
    # print(num_hit_list[:10])
    # print('thd ratio average:{}'.format(np.mean(thd_ratio_list)))

    # print('CPD accuracy:{}'.format(num_hit_cpd_all / max(1, num_match_cpd_all)))
    # print('Jeff accuracy:{}'.format(num_hit_jeff_all / max(1, num_match_jeff_all)))
    # print('Jeff_1 accuracy:{}'.format(num_hit_jeff_all_1 / max(1, num_match_jeff_all_1)))
    # print('trans time:{}, hung time:{}, cpd time:{}'.format(np.mean(time_trans_list), np.mean(time_hung_list), np.mean(time_cpd_list)))
    # print('number of gt match:{}, total count:{}'.format(np.mean(num_match_list), len(num_match_list)))
    # print('number of segment:{}'.format(np.mean(num_seg)))
    # print('number of gt label:{}, total count:{}'.format(np.mean(num_label_list), len(num_label_list)))
    # breakpoint()
    lower_bound = np.array(num_hit_all)/np.array(num_match_label_all)
    upper_bound = np.array(num_hit_upper)/np.array(num_match_label_all)
    mean_acc = [np.mean(i) for i in score_list]
    # breakpoint()
    for i,(low,up,mean) in enumerate(zip(lower_bound,upper_bound,mean_acc)):
        print(f"{i} : Upper bound:{up:.4f}, Lower bound:{low:.4f}, mean acc:{mean:.4f}")

