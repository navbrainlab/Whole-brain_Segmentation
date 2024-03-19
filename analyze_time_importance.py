import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from src.models import CNN_LSTM_Model, CNN_Model, LSTM_Model, MLP_Model
from mydataset import BMIDataset, BMIData
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import warnings
# 禁用 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)
# 定义训练函数

class2id = {'left':1,'right':0}

def gradient_hook(grad):
    print("Input tensor gradient:")
    print(grad)
def evaluate_and_prepare_cam(model, data_loader, device, weighted=True):
    model.eval()
    grad_list = {'left_T':[],'left_F':[],'right_T':[],'right_F':[]} # left:1, right:0
    input_list = {'left_T':[],'left_F':[],'right_T':[],'right_F':[]} # left:1, right:0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        # print(inputs.grad)
        inputs.requires_grad = True
        # inputs.register_hook(gradient_hook)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        scores = nn.Sigmoid()(outputs)[:,predicted]
        right_T = (labels == class2id['right']) & (predicted == labels)
        right_F = (labels == class2id['right']) & (predicted != labels)
        left_T = (labels == class2id['left']) & (predicted == labels)
        left_F = (labels == class2id['left']) & (predicted != labels)

        if torch.sum(right_T)>0:
            torch.mean(scores[right_T]).backward(retain_graph=True)
            grad_list['right_T'].append(inputs.grad[right_T])
            input_list['right_T'].append(inputs[right_T])

        if torch.sum(right_F)>0:
            torch.mean(scores[right_F]).backward(retain_graph=True)
            grad_list['right_F'].append(inputs.grad[right_F])
            input_list['right_F'].append(inputs[right_F])

        if torch.sum(left_T)>0:
            torch.mean(scores[left_T]).backward(retain_graph=True)
            grad_list['left_T'].append(inputs.grad[left_T])
            input_list['left_T'].append(inputs[left_T])

        if torch.sum(left_F)>0:
            torch.mean(scores[left_F]).backward(retain_graph=True)
            grad_list['left_F'].append(inputs.grad[left_F])
            input_list['left_F'].append(inputs[left_F])
        # else:
        #     if torch.sum(right_T) > 0:
        #         torch.mean(scores[right_T]).backward(retain_graph=True)
        #         grad_list['right_T'].append(inputs.grad[right_T]*inputs[right_T].detach())
        # 
        #     if torch.sum(right_F) > 0:
        #         torch.mean(scores[right_F]).backward(retain_graph=True)
        #         grad_list['right_F'].append(inputs.grad[right_F]*inputs[right_F].detach())
        # 
        #     if torch.sum(left_T) > 0:
        #         torch.mean(scores[left_T]).backward(retain_graph=True)
        #         grad_list['left_T'].append(inputs.grad[left_T]*inputs[left_T].detach())
        # 
        #     if torch.sum(left_F) > 0:
        #         torch.mean(scores[left_F]).backward(retain_graph=True)
        #         grad_list['left_F'].append(inputs.grad[left_F]*inputs[left_F].detach())
    T,F = [],[]
    T = T + grad_list['right_T'] + grad_list['left_T']
    F = F + grad_list['right_F'] + grad_list['left_F']
    grad_list.update({'T':T})
    grad_list.update({'F':F})

    T,F = [],[]
    T = T + input_list['right_T'] + input_list['left_T']
    F = F + input_list['right_F'] + input_list['left_F']
    input_list.update({'T':T})
    input_list.update({'F':F})
    return (input_list, grad_list)

def save_time_importance_fig(pre_cam_list,axis='time',phase='train'):
    input_list, grad_list = pre_cam_list
    for name,grad in grad_list.items():
        if len(grad)>0:
            if axis == 'time':
                inputs = torch.concatenate(input_list[name],dim=0)
                grads = torch.concatenate(grad_list[name],dim=0)
                z = grads.shape[-1]
                a_c_k = grads.sum(dim=-1) / z # norm_grads
                cam = torch.sum(nn.ReLU()(a_c_k[...,None] * inputs),dim=1)
                draw_and_save(cam.mean(dim=0)[None,...].detach().cpu().numpy(),name,phase,axis,vmin=None,vmax=None)
            else:
                inputs = torch.concatenate(input_list[name],dim=0)
                grads = torch.concatenate(grad_list[name],dim=0)
                z = grads.shape[-2]
                a_c_k = grads.sum(dim=-2) / z # norm_grads
                cam = torch.sum(nn.ReLU()(a_c_k[:,None,:] * inputs),dim=-1)
                draw_and_save(cam.mean(dim=0)[None,...].detach().cpu().numpy(),name,phase,axis,vmin=None,vmax=None)


def draw_and_save(weights, name, phase, axis, vmin, vmax):
    """
    name : T/F
    phase : train/test
    axis : time/channel
    """
    plt.figure(figsize=(10, 4))

    # plt.ylabel('Output Channel')
    if axis == 'time':
        plt.imshow(weights, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)  # 使用jet颜色映射将权重可视化
        plt.colorbar()

        plt.xlabel('time/ms')
        plt.xticks(range(0, weights.shape[-1], 40))
        plt.title(f'Weights of Input Time Sequences(train_{name})')
    else:
        plt.imshow(weights, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)  # 使用jet颜色映射将权重可视化
        plt.colorbar()

        plt.xlabel('unit/channel')
        plt.title(f'Weights of Input Units(train_{name})')
    check_dir(f'feat_weights/{phase}/{axis}')
    plt.savefig(f'feat_weights/{phase}/{axis}/{name}_{axis}_importance.png')


def check_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

if __name__ == '__main__':
    epochs = 200
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 定义模型、数据加载器、损失函数和优化器
    model = CNN_LSTM_Model(conv_kernel_size=10, cnn_input_channels=85, lstm_layers=1, lstm_embedding_dim=16, num_classes=2) # 替换为您的模型类
    # model = MLP_Model(in_dim=85 * 200, num_classes=2)
    # model = CNN_Model(conv_kernel_size=10, cnn_input_channels=85,  cnn_stride=5,num_classes=2) # 替换为您的模型类

    # model = LSTM_Model(lstm_layers=1, lstm_embedding_dim=16, num_classes=2)

    bmi_data = BMIData(electrode_path1='data/Kilosort_SingleUnits1', electrode_path2='data/Kilosort_SingleUnits2',
                       train_rate=0.8)

    train_dataset = BMIDataset(bmi_data=bmi_data, phase='train')
    test_dataset = BMIDataset(bmi_data=bmi_data, phase='test')

    train_loader = DataLoader(train_dataset, batch_size=1,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load('ckpt/lstm/198_0.98.pth',map_location='cpu'))
    model.to(device)

    for phase in ['train','test']:
        if phase == 'train':
            pre_cam_list = evaluate_and_prepare_cam(model,train_loader,device)
            save_time_importance_fig(pre_cam_list,axis='time',phase=phase)
            save_time_importance_fig(pre_cam_list, axis='channel', phase=phase)
        if phase == 'test':
            pre_cam_list = evaluate_and_prepare_cam(model,test_loader,device)
            save_time_importance_fig(pre_cam_list,axis='time',phase=phase)
            save_time_importance_fig(pre_cam_list, axis='channel', phase=phase)


