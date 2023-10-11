import pickle
import time
import copy
import tqdm

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.9996):
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

def random_object_scaling(gt_boxes, points, num_try=10):
    gt_boxes = gt_boxes.detach().cpu().numpy()
    points = points.detach().cpu().numpy()
    num_boxes = gt_boxes.shape[0]
    scale_perturb = [ 0.999, 1.001 ]
    scale_noises = np.random.uniform(scale_perturb[0], scale_perturb[1], size=[num_boxes, num_try])
    for k in range(num_boxes):
        scl_box = copy.deepcopy(gt_boxes[k])
        scl_box = scl_box.reshape(1, -1).repeat([num_try], axis=0)
        scl_box[:, 3:6] = scl_box[:, 3:6] * scale_noises[k].reshape(-1, 1).repeat([3], axis=1)

        # detect conflict
        # [num_try, N-1]
        if num_boxes > 1:
            self_mask = np.ones(num_boxes, dtype=np.bool_)
            self_mask[k] = False
            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(scl_box, gt_boxes[self_mask])
            ious = np.max(iou_matrix, axis=1)
            no_conflict_mask = (ious == 0)
            # all trys have conflict with other gts
            if no_conflict_mask.sum() == 0:
                continue

            # scale points and assign new box
            try_idx = no_conflict_mask.nonzero()[0][0]
        else:
            try_idx = 0

        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[:, 0:3],np.expand_dims(gt_boxes[k], axis=0)).squeeze(0)

        obj_points = points
        obj_center, lwh, ry = gt_boxes[k, 0:3], gt_boxes[k, 3:6], gt_boxes[k, 6]

        # relative coordinates
        obj_points[:, 0:3] -= obj_center
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), -ry).squeeze(0)
        new_lwh = lwh * scale_noises[k][try_idx]

        obj_points[:, 0:3] = obj_points[:, 0:3] * scale_noises[k][try_idx]
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), ry).squeeze(0)
        # calculate new object center to avoid object float over the road
        obj_center[2] += (new_lwh[2] - lwh[2]) / 2
        obj_points[:, 0:3] += obj_center
        points = obj_points
        gt_boxes[k, 3:6] = new_lwh

        # if enlarge boxes, remove bg points
        if scale_noises[k][try_idx] > 1:
            points_dst_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                        np.expand_dims(gt_boxes[k],
                                                                                       axis=0)).squeeze(0)

            # keep_mask = ~np.logical_xor(point_masks, points_dst_mask)
            points = points[points_dst_mask]
    
    gt_boxes = torch.from_numpy(gt_boxes).float()
    points = torch.from_numpy(points).float()
    return points, gt_boxes

def produce_ps_label(batch_dict, pred_dicts):
    batch_size = len(pred_dicts)
    gt_boxes = []
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()
        
        gt_box = np.concatenate((pred_boxes,
                                pred_labels.reshape(-1,1)), axis=1)
        gt_box = torch.from_numpy(gt_box).float().unsqueeze(0)
        gt_boxes.append(gt_box)

    batch_dict['gt_boxes'] = torch.cat(gt_boxes,dim=0).squeeze(0)
    batch_dict['points'], batch_dict['gt_boxes'][:,:7] = random_object_scaling(batch_dict['gt_boxes'][:,:7], batch_dict['points'], num_try=50)
    batch_dict['gt_boxes'] = batch_dict['gt_boxes'].unsqueeze(0).cuda()
    batch_dict['points'] = batch_dict['points'].cuda()
    return batch_dict


def eval_one_epoch(cfg, model, s_model, optimizer, dataloader, test_loader, epoch_id, logger, 
                    model_func=None, dist_test=False, save_to_file=False, result_dir=None, args=None):
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
    # eval_mem_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)
    for test_time in tqdm.tqdm(range(3), desc="training"):
        print("test time", test_time)
        for i, batch_dict in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict, t_spatial_feature_2d = model(batch_dict)

            batch_size = len(pred_dicts)
            t_spatial_feature_2d.requires_grad = True

            t_query = s_model.query_head(t_spatial_feature_2d.mean(dim=[2,3]))
            t_value = s_model.value_head(t_spatial_feature_2d.mean(dim=[2,3]))
            batch_dict = produce_ps_label(batch_dict=batch_dict, pred_dicts=pred_dicts)
            loss_model, s_tb_dict, s_disp_dict, s_spatial_feature_2d = model_func(s_model, batch_dict)

            # s_spatial_feature_2d = max_pool(s_spatial_feature_2d).contiguous.view(1,-1)
            s_query = s_model.query_head(s_spatial_feature_2d.mean(dim=[2,3]))
            s_value = s_model.value_head(s_spatial_feature_2d.mean(dim=[2,3]))
            s_model.mem_bank = s_model.memory_update(s_model.mem_bank,
                                                     t_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                                     t_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                     )
            mem_s_query  = s_model.memory_read(s_model.mem_bank,
                                                s_query.contiguous().unsqueeze(-1).unsqueeze(-1), 
                                                s_value.contiguous().unsqueeze(-1).unsqueeze(-1),
                                                )
            loss_mem = s_model.get_mem_loss(s_query, s_spatial_feature_2d, mem_s_query.squeeze(-1).squeeze(-1), s_value,t_spatial_feature_2d, t_value, s_model.mem_bank)
            disp_dict = {}
            loss = loss_mem + loss_model
            loss.backward()
            optimizer.step()
            new_teacher_dict = update_teacher_model(s_model, model, keep_rate=0.99999)
            model.load_state_dict(new_teacher_dict)
        
        eval_mem_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)
            
    eval_mem_epoch(cfg, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=result_dir, save_to_file=save_to_file, args=args)

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

def eval_mem_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, args=None):
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

    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict, _ = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
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
