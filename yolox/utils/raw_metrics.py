import cv2
import os
from tqdm import tqdm

import numpy as np
import torch

from xml.etree.ElementTree import parse
from collections import defaultdict

from yolox.data.datasets import VOC_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess
from yolox.data.data_augment import preproc as preprocess

from mlops_utils.data import get_data_from_voc, make_batchwise_format
from mlops_utils.model import Evaluator

def imagewise_pred(self, origin_img, model, model_cls_names, score_thr, nms_thr):
        input_shape = "544,960"
        input_shape = tuple(map(int, input_shape.split(',')))
        img, ratio = preprocess(origin_img, input_shape)
    
        model = model.eval()        
        output = model(torch.tensor(img).view(1, 3, input_shape[0], input_shape[1]).cuda()).detach()
        output = torch.cat([output[..., :4], output[..., 4:].sigmoid()], dim=-1)
        output = output.cpu().numpy()

        predictions = demo_postprocess(output, input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

        if dets is None:
            return None
        else:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        scores = []
        labels = []
        bboxes = []
        try:
            for i in range(dets.shape[0]):
                scores.append(final_scores[i])
                labels.append(model_cls_names[int(final_cls_inds[i])])
                bboxes.append(list(final_boxes[i]))

            results = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }
            
        return results
        

def batchwise_pred(batchwise_img_list, model, model_cls_names, nms_thr, score_thr):
    pred_batchwise_bbox_list = []
    pred_batchwise_label_list = []
    pred_batchwise_score_list = []
    for batchwise_img in tqdm(batchwise_img_list):
        pred_batchwise_bbox = []
        pred_batchwise_label = []
        pred_batchwise_score = []
        for img in batchwise_img:
            results = imagewise_pred(
                model=model, 
                model_cls_names=model_cls_names,
                origin_img=img,
                nms_thr=nms_thr,
                score_thr=score_thr,
            )
            try:
                pred_batchwise_bbox.append(results["bboxes"])
                pred_batchwise_label.append(results["labels"])
                pred_batchwise_score.append(results["scores"])
            except:
                # 객체가 없는 경우
                pred_batchwise_bbox.append([])
                pred_batchwise_label.append([])
                pred_batchwise_score.append([])

        pred_batchwise_bbox_list.append(np.array(pred_batchwise_bbox))
        pred_batchwise_label_list.append(np.array(pred_batchwise_label))
        pred_batchwise_score_list.append(np.array(pred_batchwise_score))

    return pred_batchwise_bbox_list, pred_batchwise_label_list, pred_batchwise_score_list




def get_raw_metrics(model, nms_thr, score_thr, voc_data_dir="datasets/VOCdevkit")
    model_cls_names = list(VOC_CLASSES)

    test_name_list, actual_bbox_list, actual_label_list, num_obj_dict, img_arr_list = get_data_from_voc(voc_data_dir=voc_data_dir, model_cls_names=model_cls_names, is_test=True)

    actual_batchwise_bbox_list = make_batchwise_format(actual_bbox_list)
    actual_batchwise_label_list = make_batchwise_format(actual_label_list)
    batchwise_img_list = make_batchwise_format(img_arr_list)

    pred_batchwise_bbox_list, pred_batchwise_label_list, pred_batchwise_score_list = batchwise_pred(
        batchwise_img_list=batchwise_img_list, 
        model=model, 
        model_cls_names=model_cls_names, 
        nms_thr=nms_thr, 
        score_thr=score_thr
    )

    evaluator = Evaluator()

    conf_mat_results = evaluator.conf_mat_eval(
        total_pos_class_dict=num_obj_dict,
        actual_batchwise_bbox_list=actual_batchwise_bbox_list, 
        actual_batchwise_label_list=actual_batchwise_label_list, 
        pred_batchwise_bbox_list=pred_batchwise_bbox_list, 
        pred_batchwise_label_list=pred_batchwise_label_list,
        iou_thr=0.5
    )

    return conf_mat_results