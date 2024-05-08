import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import os
import glob
import sys
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
sys.path.append('/mnt/c/gaochao/CODE/NeuronTrack/fDNC_Neuron_ID')
from src.DGCNN_model import *

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def find_match(label2, label1):
    # label 1 and label 2 is the label for neurons in two worms.
    if len(label1) == 0 or len(label2) == 0:
        return []
    pt1_dict = dict()
    # build a dict of neurons in worm1
    label1_indices = np.where(label1 >= 0)[0]
    for idx1 in label1_indices:
        pt1_dict[label1[idx1]] = idx1
    # search label2 in label 1 dict
    match = list()
    unlabel = list()
    label2_indices = np.where(label2 >= -1)[0]
    for idx2 in label2_indices:
        if label2[idx2] in pt1_dict:
            match.append([idx2, pt1_dict[label2[idx2]]])
        else:
            unlabel.append(idx2)
    return np.array(match), np.array(unlabel)


class neuron_data_pytorch(Dataset):
    """
    This class is to compile neuron data from different worms
    """
    def __init__(self, path, batch_sz, shuffle, rotate=False, mode='all', ref_idx=0, show_name=False, shuffle_pt = True, tmp_path= None):
        """
        Initialize parameters.
        :param path: the path for all worms
        :param batch_sz: batch_size
        :param shuffle : whether to shuffle the data
        """
        self.path = path
        self.mode = mode
        self.batch_sz = batch_sz
        self.shuffle = shuffle

        self.rotate = rotate
        self.ref_idx = ref_idx
        self.show_name = show_name
        self.shuffle_pt = shuffle_pt

        # set the temp_plate
        if tmp_path is not None:
            # set the ref_idx to 0
            self.ref_idx = 0
            self.tmp_path = tmp_path
            self.load_path(path, batch_sz-1)
        else:
            self.tmp_path = tmp_path
            self.load_path(path, batch_sz)

    def load_path(self, path, batch_sz):
        """
        This function get the folder names + file names(volume) together. Set the index for further use
        :param batch_sz: batch size
        :return:
        """
        if self.mode == 'all':
            self.folders = glob.glob(os.path.join(path, '*/'))
        elif self.mode == 'real':
            self.folders = glob.glob(os.path.join(path, 'real_*/'))
        elif self.mode == 'syn':
            self.folders = glob.glob(os.path.join(path, 'syn_*/'))

        # files in each folder is a list
        all_files = list()
        bundle_list = list()
        num_data = 0

        for folder_idx, folder in enumerate(self.folders):
            if self.mode == 'all':
                volume_list = glob.glob1(folder, '*.npy')
            elif self.mode == 'real':
                volume_list = glob.glob1(folder, 'real_*.npy')
            elif self.mode == 'syn':
                volume_list = glob.glob1(folder, 'syn_*.npy')

            num_volume = len(volume_list)
            num_data += num_volume
            all_files.append(volume_list)

            if batch_sz > num_volume:
                bundle_list.append([folder_idx, 0, num_volume])
            else:
                for i in range(0, num_volume, batch_sz):
                    end_idx = i + batch_sz
                    if end_idx > num_volume:
                        end_idx = num_volume
                        start_idx = num_volume - batch_sz
                    else:
                        start_idx = i

                    bundle_list.append([folder_idx, start_idx, end_idx])

        self.all_files = all_files
        self.bundle_list = bundle_list
        self.batch_num = len(bundle_list)
        print('total volumes:{}'.format(num_data))
        if self.shuffle:
            self.shuffle_batch()

    def __len__(self):
        return self.batch_num

    def shuffle_batch(self):
        """
        This function shuffles every element in all_files list(volume) and bundle list(order of batch)
        :return:
        """
        for volumes_list in self.all_files:
            np.random.shuffle(volumes_list)

        np.random.shuffle(self.bundle_list)

    def __getitem__(self, item):
        return self.bundle_list[item]


    def load_pt(self, pt_name):
        pt = np.load(pt_name)
        if self.shuffle_pt:
            np.random.shuffle(pt)
        if self.rotate:
            theta = np.random.rand(1)[0] * 2 * np.pi
            r_m = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], \
                   [0, 0, 1]]
            pt[:, :3] = np.matmul(pt[:, :3], r_m)
        return pt

    def custom_collate_fn(self, bundle):
        bundle = bundle[0]
        pt_batch = list()
        label_batch = list()
        pt_name_list = list()

        if self.tmp_path is not None:
            if self.show_name:
                pt_name_list.append(self.tmp_path)
            pt = self.load_pt(self.tmp_path)
            pt_batch.append(pt[:, :3])
            label_batch.append(pt[:, 3])

        for volume_idx in range(bundle[1], bundle[2]):
            pt_name = os.path.join(self.folders[bundle[0]], self.all_files[bundle[0]][volume_idx])
            if self.show_name:
                pt_name_list.append(pt_name)

            pt = self.load_pt(pt_name)
            pt_batch.append(pt[:, :3])
            label_batch.append(pt[:, 3])

        match_dict = dict()

        ref_i = self.ref_idx
        for i in range(len(label_batch)):
            match_dict[i], match_dict['unlabel_{}'.format(i)] = find_match(label_batch[i], label_batch[ref_i])
            # get the unlabelled neuron
            #match_dict['unlabel_{}'.format(i)] = np.where(label_batch[i] == -1)[0]
            # get the outlier neuron
            match_dict['outlier_{}'.format(i)] = np.where(label_batch[i] == -2)[0]

        data_batch = dict()
        data_batch['pt_batch'] = pt_batch
        data_batch['match_dict'] = match_dict
        data_batch['pt_label'] = label_batch
        data_batch['ref_i'] = ref_i
        if self.show_name:
            data_batch['pt_names'] = pt_name_list
        return data_batch

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        params = {
            'args': dict(input_dim=self.input_dim,
                         n_hidden=self.n_hidden,
                         n_layer=self.n_layer,
                         cuda=self.cuda),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class TransformerDecoder(nn.TransformerDecoder):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        output_list = []
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            output_list.append(output)

        if self.norm is not None:
            output =  [self.norm[i](o) for i,o in enumerate(output_list)]

        return output

class TransformerEncoder(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal


        output_list = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            output_list.append(output)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = [self.norm[i](o) for i,o in enumerate(output_list)]

        return output







class NIT_Registration(nn.Module):
    """ Neuron Id transformer for registration between two datasets:
        - Transformer
    """

    def __init__(self, input_dim, n_hidden, n_layer=6, cuda=True, p_rotate=False, feat_trans=False,
                    use_embedding = True,
                    drop_neuron_ratio = 0.3,
                ):
        super(NIT_Registration, self).__init__()
 
        self.use_embedding = use_embedding
        # breakpoint()
        self.drop_neuron_ratio = drop_neuron_ratio

        self.encoder_layer_num = 6
        self.decoder_layer_num = 6
        encoder_norm = _get_clones(nn.LayerNorm(n_hidden),self.encoder_layer_num)
        decoder_norm = _get_clones(nn.LayerNorm(n_hidden),self.decoder_layer_num)
        self.enc_l = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=8, norm_first=True, dim_feedforward=1024)
        self.encoder_model = TransformerEncoder(self.enc_l, self.encoder_layer_num, norm = encoder_norm)

        # self.encoder_model = nn.Identity()
        self.dec_l = nn.TransformerDecoderLayer(d_model=n_hidden, nhead=8, norm_first=True, dim_feedforward=1024)
        self.decoder_model = TransformerDecoder(self.dec_l, self.decoder_layer_num, norm = decoder_norm)

        self.device = torch.device("cuda:0" if cuda else "cpu")
        #----------------------add-------------------------------
        self.xyz_ptf = nn.Sequential(
            nn.Conv1d(3, n_hidden, 1),
            nn.BatchNorm1d(n_hidden)
        )

        
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls = nn.Embedding(1,n_hidden)
        self.decode_xyz = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,3)
        )

        self._reset_parameters()

    def to_input_tensor(self, pts, pad_pt=[0, 0, 0]):
        sents_padded = []
        max_len = max(len(s) for s in pts)
        for s in pts:
            padded = [pad_pt] * max_len
            padded[:len(s)] = s
            sents_padded.append(padded)
        #sents_var = torch.tensor(sents_padded, dtype=torch.long, device=self.device)
        sents_var = torch.tensor(np.array(sents_padded), dtype=torch.float, device=self.device)
        return sents_var #torch.t(sents_var)

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float).bool()
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = True
        return enc_masks.to(self.device)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, pts_padded, pts_length, ref_idx):
        xyz_pts_proj = self.xyz_ptf(pts_padded.transpose(2, 1))
        xyz_pts_proj = xyz_pts_proj.transpose(2,1)
        mask = self.generate_sent_masks(xyz_pts_proj, pts_length)
        decoded_to_return, temp_cls_token = self.for_encode(xyz_pts_proj, pts_length, mask, ref_idx)
        to_return = ([],[])
        to_return[0].extend(decoded_to_return[:6])
        to_return[1].append(temp_cls_token)
        return to_return


    def for_encode(self, pts_proj, pts_length, mask, ref_idx):
        cls_mask = torch.zeros(64,1,device=self.device).bool()

        cls_token = torch.repeat_interleave(self.cls.weight,64,0)

        cat_cls_mask = torch.concatenate([cls_mask, mask],dim=1)
        cat_cls_token = torch.concat([cls_token[:,None,:], pts_proj],dim=1)

        pts_encode = self.encoder_model(cat_cls_token.transpose(dim0=0, dim1=1),src_key_padding_mask=cat_cls_mask)
        temp_cls_token = [i[0,:,:] for i in pts_encode][-1][ref_idx]
        pts_encode = [i[1:,:,:] for i in pts_encode]

        ref_pts_proj = pts_encode[-1][:pts_length[ref_idx], ref_idx:ref_idx+1, :]
        encode_layers_ref_pts_proj = [p[:pts_length[ref_idx], ref_idx:ref_idx+1, :] for p in pts_encode]

        tgt_input = pts_encode[-1]
        tgt_mask = mask
        ref_pts_proj = torch.repeat_interleave(ref_pts_proj, len(pts_length), dim=1)
        encode_layers_ref_pts_proj = [torch.repeat_interleave(i, len(pts_length), dim=1) for i in encode_layers_ref_pts_proj]

        mem_mask = None
        
        pts_decoded = self.decoder_model(tgt=tgt_input, tgt_key_padding_mask=tgt_mask, 
                                         memory=ref_pts_proj, memory_key_padding_mask=mem_mask)

        decoded_to_return = [torch.concatenate([ref_pts_proj,pts_d],dim=0).transpose(1,0) for pts_d in pts_decoded]
        return (decoded_to_return, temp_cls_token)

    def generate_mem_mask(self, num_rows, num_cols, r, device):
        vector = torch.zeros(num_rows, num_cols,device=device)
        num_ones = int(num_cols * r)

        # 在每一行随机选择 num_ones 个位置设置为 1
        for i in range(num_rows):
            indices = torch.randperm(num_cols)[:num_ones]
            vector[i, indices] = 1
    
        return vector.bool()

    def forward(self, pts, match_dict=None, ref_idx=0, mode='train'):
        """ Take a mini-batch of "source" and "target" points, compute the encoded hidden code for each
        neurons.
        @param pts1 (List[List[x, y, z]]): list of source points set
        @param pts2 (List[List[x, y, z]]): list of target points set

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        self.match_dict = match_dict
        # Compute sentence lengths
        pts_lengths = [len(s) for s in pts]
        # pts_padded should be of dimension (
        pts_padded = self.to_input_tensor(pts)
        # mov_emb, ref_emb = self.encode(pts_padded, pts_lengths, ref_idx)


        pts_encode, temp_cls_token = self.encode(pts_padded, pts_lengths, ref_idx)
        # encoded_pair_contrastive.extend(pts_encode)
        # pts_encode = encoded_pair_contrastive
        # if self.training:
        # ref_emb_list = [pts_encode[i][:, :pts_lengths[ref_idx], :] for i in range(len(pts_encode))]
        # mov_emb_list = [pts_encode[i][:, pts_lengths[ref_idx]:, :] for i in range(len(pts_encode))]

        ref_emb_list = [F.normalize(pts_encode[i], p=2, dim=2, eps=1e-12, out=None)[:, :pts_lengths[ref_idx], :] for i in range(len(pts_encode))] # 记得normalize之前加入outlier embedding
        mov_emb_list = [F.normalize(pts_encode[i], p=2, dim=2, eps=1e-12, out=None)[:, pts_lengths[ref_idx]:, :] for i in range(len(pts_encode))]

        # sim_m = torch.mean(torch.stack([torch.bmm(mov_emb, ref_emb.transpose(dim0=1, dim1=2)) for mov_emb,ref_emb in zip(mov_emb_list, ref_emb_list)]),dim=0)
        # # the outlier of mov node
        # mov_outlier = self.fc_outlier(torch.mean(torch.stack(mov_emb_list),dim=0))
        logit_scale = self.logit_scale.exp()
        layers_sim_m = list(torch.stack([logit_scale * torch.bmm(mov_emb, ref_emb.transpose(dim0=1, dim1=2)) for mov_emb,ref_emb in zip(mov_emb_list, ref_emb_list)]))
        # the outlier of mov node
        # layers_mov_outlier = self.fc_outlier(torch.stack(mov_emb_list))
        # layers_sim_m = [torch.cat((sim_m, mov_outlier), dim=2) for sim_m, mov_outlier in zip(layers_sim_m, layers_mov_outlier)]
        
        # p_m = [F.log_softmax(sim_m, dim=2) for sim_m in layers_sim_m]
        p_m_exp = [F.softmax(sim_m, dim=2) for sim_m in layers_sim_m]
        p_m_exp_alongy = [F.softmax(sim_m, dim=1) for sim_m in layers_sim_m]


        # p_m_exp = [i+j for i,j in zip(p_m_exp,p_m_exp_alongy)]


        batch_sz = mov_emb_list[0].size(0)
        loss = 0
        num_pt = 0
        loss_entropy = 0
        mse_loss = 0
        num_unlabel = 0
        output_pairs = dict()
        loss_dict = dict()
        
        if (mode == 'train') or (mode == 'all'):
            # loss_dict['xyz_loss'] = F.mse_loss(self.fc_xyz(pts_decoded4xyz), pts_xyz)
            for i_w in range(batch_sz):
                # loss for labelled neurons.
                match = match_dict[i_w]
                if len(match) > 0:
                    match_mov = match[:, 0]
                    match_ref = match[:, 1]
                    # along x axis,
                    log_p = torch.stack([p[i_w, match_mov, match_ref] for p in p_m_exp])## 每个test neuron与真实对应temp neuron的匹配概率
                    log_p_neg = torch.stack([p[i_w, match_mov, :] for p in p_m_exp])
                    loss += (log_p_neg.sum() - 2*log_p.sum())/torch.prod(torch.tensor(log_p.shape))

                    # -----------------------------along y axis
                    log_p = torch.stack([p[i_w, match_mov, match_ref] for p in p_m_exp_alongy])## 每个test neuron与真实对应temp neuron的匹配概率      
                    log_p_neg = torch.stack([p[i_w, :, match_ref] for p in p_m_exp_alongy])
                    loss += (log_p_neg.sum() - 2*log_p.sum())/torch.prod(torch.tensor(log_p.shape))

                    # mse loss 
                    mse = F.mse_loss(mov_emb_list[-1][i_w,match_mov,:], ref_emb_list[-1][i_w, match_ref,:],reduction='mean')
                    loss += mse

                    # fintune decode xyz loss
                    layers_emb = torch.stack([i[i_w,match_mov] for i in mov_emb_list],dim=0) + temp_cls_token[0][None,None,:]
                    p_xyz = self.decode_xyz(layers_emb)
                    xyz_label = pts_padded[i_w, match_ref,:].repeat(6,1,1)
                    xyz_loss = F.mse_loss(p_xyz, xyz_label)
                    loss += xyz_loss

                    num_pt += len(match_mov)
                # loss for outliers.
                outlier_list = self.match_dict['outlier_{}'.format(i_w)]
                # if len(outlier_list) > 0:
                #     log_p_outlier = torch.stack([p[i_w, outlier_list, -1] for p in p_m_exp])
                #     loss -= 1*log_p_outlier.sum()/torch.prod(torch.tensor(log_p_outlier.shape))
                #     num_pt += len(outlier_list)

                # Entropy loss for unlabelled neurons.
                unlabel_list = self.match_dict['unlabel_{}'.format(i_w)]
                if len(unlabel_list) > 0:
                    ## 2024-2-7 unlabel 直接忽略
                    # loss_entropy_cur = p_m[i_w, unlabel_list, :] * p_m_exp[i_w, unlabel_list, :]
                    # loss_entropy_cur = torch.stack([p[i_w, unlabel_list, :] * p_exp[i_w, unlabel_list, :] for p,p_exp in zip(p_m, p_m_exp)])
                    # loss_entropy -= loss_entropy_cur.sum()
                    num_unlabel += len(unlabel_list)

        elif (mode == 'eval') or (mode == 'all'):
            # breakpoint()
            p_m_exp.append(torch.mean(torch.stack(p_m_exp),dim=0))
            # ensemble_p_m_exp = p_m_exp[0]*0.1 + p_m_exp[1]*0.1+ p_m_exp[2]*0.11+ p_m_exp[3]*0.12+ p_m_exp[4]*0.13 + p_m_exp[5]*0.15
            # p_m_exp.append(ensemble_p_m_exp)

            # p_m_exp.extend(p_m)
            output_pairs['p_m'] = p_m_exp
            # weight = [1,5,6,6,6,6, 1,1,1,1,0,0]
            # weighted_ensemble = torch.sum(torch.stack([i*w for i,w in zip(layers_sim_m,weight)],dim=0),dim=0)
            layers_sim_m.append(torch.mean(torch.stack(layers_sim_m[2:]),dim=0))
            # layers_sim_m.append(weighted_ensemble)
            output_pairs['p_m'] = layers_sim_m
            save_temp_test_embed = [{'temp':ref_emb[0] ,'test':mov_emb} for mov_emb,ref_emb in zip(mov_emb_list, ref_emb_list)]
            output_pairs['save_embedd'] = save_temp_test_embed
            output_pairs['logits_scale'] = logit_scale               
            # choose the matched pairs for worm1
            paired_idx = torch.argmax(p_m_exp[-1], dim=1)
            output_pairs['paired_idx'] = paired_idx
            # TODO:
            # pick the maxima value

            # output_pairs['raw_product'] = torch.mean(torch.stack(layers_sim_m),dim=0)
        loss_dict['loss'] = loss
        loss_dict['num'] = num_pt if num_pt else 1
        loss_dict['loss_entropy'] = loss_entropy
        loss_dict['num_unlabel'] = num_unlabel if num_unlabel else 1

        # loss_dict['reg_stn'] = self.point_f.reg_stn
        # loss_dict['reg_fstn'] = self.point_f.reg_fstn

        return loss_dict, output_pairs
    
    def get_dist_weight(self, p_m, pts, match_mov, match_ref):
        pred_id = torch.argmax(p_m,dim=-1)[match_mov]
        pred_pts_pos = pts[pred_id]
        true_pts_pos = pts[match_ref]
        diff = pred_pts_pos - true_pts_pos
        weight = torch.sqrt(diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2)
        weight = torch.tensor(500).pow(weight)
        return weight

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        params = {
            'state_dict': self.state_dict()
        }

        torch.save(params, path)