import torch
from torch import nn
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn import functional as F
from .memory_bank import Memory_trans_update, Memory_trans_read


class SECONDNetIoU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == "BaseBEVBackbone":
                spatial_features_2d = batch_dict['spatial_features_2d']

        if self.training:
            weights = batch_dict.get('SEP_LOSS_WEIGHTS', None)
            loss, tb_dict, disp_dict = self.get_training_loss(weights)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            return pred_dicts, recall_dicts, spatial_features_2d

    def get_training_loss(self, weights=None):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss(weights)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        iou_weight = 1.0
        if weights is not None:
            iou_weight = weights[-1]

        loss = loss_rpn + iou_weight * loss_rcnn
        return loss, tb_dict, disp_dict


class SECONDNetIoUSTU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        mem_items = 10240
        mem_features = 512
        self.mem_bank = F.normalize(torch.rand((mem_items, mem_features), dtype=torch.float), dim=1).cuda()
        self.memory_update = Memory_trans_update(memory_size=mem_items, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1)
        self.memory_read = Memory_trans_read()
        dim_in = 512
        dim_out = 512
        feat_dim = 512
        self.query_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )
        self.value_head = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, feat_dim)
            )

    def get_score(self, mem_bank, query, items=None):
        bs, h, w, d = query.size()
        m, d = mem_bank.size()
        score = torch.matmul(query.float(), torch.t(mem_bank).float())# b X h X w X m
        score = score.view(bs*h*w, m) # 300x512
        score_memory = F.softmax(score,dim=1) # 300x512

        _, top_neg_idx = torch.topk(score_memory, items, dim=1, largest=False)

        neg_logits = torch.gather(score, 1, top_neg_idx)

        return neg_logits

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == "BaseBEVBackbone":
                spatial_features_2d = batch_dict['spatial_features_2d']
                # self.s_query = self.query_head(spatial_features_2d)
                # self.s_value = self.value_head(spatial_features_2d)

        if self.training:
            weights = batch_dict.get('SEP_LOSS_WEIGHTS', None)
            loss, tb_dict, disp_dict = self.get_training_loss(weights)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict, spatial_features_2d
        else:
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, weights=None):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss(weights)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        iou_weight = 1.0
        if weights is not None:
            iou_weight = weights[-1]

        loss = loss_rpn + iou_weight * loss_rcnn
        return loss, tb_dict, disp_dict
    
    
    def get_mem_loss(self, s_query, s_box_feat, mem_s_query, s_value, t_box_feat, t_value, mem_bank, temperature=0.07, base_temperature=0.07):
        
        batch_size, dim = s_query.shape
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        anchor_feat = F.normalize(s_query, dim=1)
        contrast_feat = F.normalize(mem_s_query, dim=1)

        logits = torch.div(torch.matmul(anchor_feat, contrast_feat.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        sm_logits = logits - logits_max.detach()

        mem_query = s_box_feat.mean(dim=[2, 3]).contiguous().unsqueeze(-1).unsqueeze(-1).permute(0,2,3,1).detach()
        sm_neg_logits = self.get_score(mem_bank, mem_query, items=5)

        s_all_logits = torch.exp(torch.cat((sm_logits, sm_neg_logits), dim=1))
        log_prob = sm_logits - torch.log(s_all_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos 

        if torch.isnan(loss.mean()):
            loss = loss*0
            
        return loss.mean()
