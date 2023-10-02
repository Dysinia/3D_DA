import pickle
import time

import numpy as np
import torch
import tqdm

from collections import OrdereDict
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.dsnorm import set_ds_target


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    student_model_dict = model_student.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

def eval_one_epoch(cfg, model, s_model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, args=None, 
                   model_func=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    s_model.train()

    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    optimizer = torch.optim.SGD(s_model.parameters(), lr=0.01)
    for i, batch_dict in enumerate(dataloader):
        optimizer = torch.zero_grad()
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict, t_spatial_feature_2d = model(batch_dict)
        t_query = s_model.query_head(t_spatial_feature_2d)
        t_value = s_model.value_head(t_spatial_feature_2d)
        ## here the batch dict should be updated
        ## batch_dict['gt_boxes'] = pred_dicts['pred_boxes']
        loss_model, s_tb_dict, s_disp_dict, s_spatial_feature_2d = model_func(s_model, batch)
        s_query = s_model.query_head(s_spatial_feature_2d)
        s_value = s_model.value_head(s_spatial_feature_2d)
        s_model.mem_bank = s_model.memory_update(s_model.mem_bank,
                                                 t_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                                 t_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                 )
        mem_s_query  = s_model.memory_read(s_model.mem_bank,
                                            s_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                            s_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                            )
        loss_mem = s_model.mem_contrastive_loss(s_query, s_spatial_feature_2d, mem_s_query.squeeze(-1).squeeze(-1), s_value,t_spatial_feature_2d, t_value, self.mem_bank)
        disp_dict = {}
        loss = loss_mem + loss_model
        loss.backward()
        optimizer.step()
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        new_teacher_dict = update_teacher_model(model, s_model, keep_rate=0.9)
        model.load_state_dict(new_teacher_dict)
        det_annos += annos
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict['eval_avg_pred_bboxes'] = total_pred_objects / max(1, len(det_annos))

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
