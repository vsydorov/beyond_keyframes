import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, TypedDict
from collections import namedtuple
from pathlib import Path

from thes.tools import snippets
from thes.tubes.routines import numpy_iou
from thes.data.external_dataset import (
        DALY_vid)


log = logging.getLogger(__name__)

BoxLTRD = namedtuple('BoxLTRD', 'l t r d')
Flat_annotation = namedtuple('Flat_annotation', 'id diff l t r d')
Flat_detection = namedtuple('Flat_detection', 'id id_box score l t r d')


class AP_fgt_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, anno_id
    obj: np.ndarray  # LTRD box
    diff: bool


class AP_fdet_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, det_id
    obj: np.ndarray  # LTRD box
    score: float


VOClike_object = TypedDict('VOClike_object', {
    'name': str,
    'difficult': bool,
    'box': BoxLTRD
})

VOClike_image_annotation = TypedDict('VOClike_image_annotation', {
    'filepath': Path,
    'frameNumber': int,  # if '-1' we treat filename is image, otherwise as video
    'size_WHD': Tuple[int, int, int],
    'objects': List[VOClike_object]
})


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t_ in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t_) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t_])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def box_area(box: BoxLTRD) -> float:
    return float((box.d-box.t+1)*(box.r-box.l+1))


def box_intersection(
        bb1: BoxLTRD,
        bb2: BoxLTRD
            ) -> BoxLTRD:
    """ Intersection of two bbs
    Returned bb has same type as first argument"""

    return BoxLTRD(
            l=max(bb1.l, bb2.l),
            t=max(bb1.t, bb2.t),
            r=min(bb1.r, bb2.r),
            d=min(bb1.d, bb2.d))


def t_box_iou(
        bb1: BoxLTRD,
        bb2: BoxLTRD
            ) -> float:

    inter = box_intersection(bb1, bb2)
    if (inter.t >= inter.d) or (inter.l >= inter.r):
        return 0.0  # no intersection
    else:
        intersection_area = box_area(inter)
        union_area = box_area(bb1) + box_area(bb2) - intersection_area
        return intersection_area/union_area


def oldcode_evaluate_voc_detections(
        annotation_list: List[VOClike_image_annotation],
        all_boxes: List[Dict[str, np.array]],
        object_classes: List[str],
        iou_thresh,
        use_07_metric,
        use_diff
            ) -> Dict[str, float]:
    """
    Params:
    use_07_metric: If True, will evaluate AP over 11 points, like in
        VOC2007 evaluation code
    use_diff: If True, will eval difficult proposals in the same way as
        real ones
    """

    assert len(annotation_list) == len(all_boxes)

    ap_per_cls: Dict[str, float] = {}
    for obj_cls in object_classes:
        # Extract GT annotation_list (id diff l t r d)
        flat_annotations: List[Flat_annotation] = []
        for ind, image_anno in enumerate(annotation_list):
            for obj in image_anno['objects']:
                if obj['name'] == obj_cls:
                    flat_annotations.append(Flat_annotation(
                        ind, obj['difficult'], *obj['box']))
        # Extract detections (id id_box score l t r d)
        flat_detections: List[Flat_detection] = []
        for id, dets_per_cls in enumerate(all_boxes):
            for id_box, det in enumerate(dets_per_cls[obj_cls]):
                flat_detections.append(
                        Flat_detection(id, id_box, det[4], *det[:4]))

        # Prepare
        detection_matched = np.zeros(len(flat_detections), dtype=bool)
        gt_already_matched = np.zeros(len(flat_annotations), dtype=bool)
        gt_assignment = {}  # type: Dict[int, List[int]]
        for i, fa in enumerate(flat_annotations):
            gt_assignment.setdefault(fa.id, []).append(i)
        # Provenance
        detection_matched_to_which_gt = np.ones(
                len(flat_detections), dtype=int)*-1

        # VOC2007 preparation
        nd = len(flat_detections)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        if use_diff:
            npos = len(flat_annotations)
        else:
            npos = len([x for x in flat_annotations if not x.diff])

        # Go through ordered detections
        detection_scores = np.array([x.score for x in flat_detections])
        detection_scores = detection_scores.round(3)
        sorted_inds = np.argsort(-detection_scores)
        for d, detection_ind in enumerate(sorted_inds):
            detection: Flat_detection = flat_detections[detection_ind]

            # Check available GTs
            gt_ids_that_share_image = \
                    gt_assignment.get(detection.id, [])  # type: List[int]
            if not len(gt_ids_that_share_image):
                fp[d] = 1
                continue

            # Compute IOUs
            det_box = BoxLTRD(*(np.array(detection[3:]).round(1)+1))
            gt_boxes: List[BoxLTRD] = [BoxLTRD(*(np.array(
                flat_annotations[gt_ind][2:]).round(1)+1))
                for gt_ind in gt_ids_that_share_image]
            iou_coverages = [t_box_iou(det_box, gtb) for gtb in gt_boxes]

            max_coverage_id = np.argmax(iou_coverages)
            max_coverage = iou_coverages[max_coverage_id]
            gt_id = gt_ids_that_share_image[max_coverage_id]

            # Mirroring voc_eval
            if max_coverage > iou_thresh:
                if (not use_diff) and flat_annotations[gt_id].diff:
                    continue
                if not gt_already_matched[gt_id]:
                    tp[d] = 1
                    detection_matched[detection_ind] = True
                    gt_already_matched[gt_id] = True
                    detection_matched_to_which_gt[detection_ind] = gt_id
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        ap_per_cls[obj_cls] = ap
        # sklearn version does not work, since it assumes 'recall=1'
        # ap = sklearn.metrics.average_precision_score(detection_matched, detection_scores)
    return ap_per_cls


def datalist_to_voclike(object_names, datalist):
    voclike_annotation_list = []
    for dl_item in datalist:
        objects = []
        for dl_anno in dl_item['annotations']:
            name = object_names[dl_anno['category_id']]
            box = BoxLTRD(*dl_anno['bbox'])
            difficult = dl_anno.get('is_occluded', False)
            o = VOClike_object(name=name, difficult=difficult, box=box)
            objects.append(o)
        filepath = dl_item['video_path']
        frameNumber = dl_item['video_frame_number']
        size_WHD = (dl_item['width'], dl_item['height'], 3)
        voclike_annotation = VOClike_image_annotation(
                filepath=filepath, frameNumber=frameNumber,
                size_WHD=size_WHD, objects=objects)
        voclike_annotation_list.append(voclike_annotation)
    return voclike_annotation_list


def voclike_legacy_evaluation(object_names, datalist, predicted_datalist):
    """
    This is the evaluation code I used 2 years ago
    """
    # // Transform to legacy data format
    # //// GroundTruth
    voclike_annotation_list: List[VOClike_image_annotation] = \
            datalist_to_voclike(object_names, datalist)
    # //// Detections
    all_boxes: List[Dict[str, np.array]] = []
    for dl_item, pred_item in zip(datalist, predicted_datalist):
        pred_boxes = pred_item.pred_boxes.tensor.numpy()
        scores = pred_item.scores.numpy()
        pred_classes = pred_item.pred_classes.numpy()
        dets = {}
        for b, s, c_ind in zip(pred_boxes, scores, pred_classes):
            cls = object_names[c_ind]
            dets.setdefault(cls, []).append(np.r_[b, s])
        dets = {k: np.vstack(v) for k, v in dets.items()}
        dets = {k: dets.get(k, np.array([])) for k in object_names}
        all_boxes.append(dets)

    use_07_metric = False
    use_diff = False
    iou_thresh = 0.5
    object_classes = object_names
    ap_per_cls: Dict[str, float] = oldcode_evaluate_voc_detections(
            voclike_annotation_list,
            all_boxes,
            object_classes,
            iou_thresh, use_07_metric, use_diff)

    # Results printed nicely via pd.Series
    x = pd.Series(ap_per_cls)*100
    x.loc['AVERAGE'] = x.mean()
    table = snippets.string_table(
            np.array(x.reset_index()),
            header=['Object', 'AP'],
            col_formats=['{}', '{:.2f}'], pad=2)
    log.info(f'AP@{iou_thresh:.3f}:\n{table}')


def evaluate_voc_detections_v2(
        fgts: List[AP_fgt_framebox],
        fdets: List[AP_fdet_framebox],
        iou_thresh,
        use_07_metric,
        use_diff
            ) -> float:
    """
    Params:
    use_07_metric: If True, will evaluate AP over 11 points, like in
        VOC2007 evaluation code
    use_diff: If True, will eval difficult proposals in the same way as
        real ones
    """
    # Group fdets belonging to same frame, assign to fgts
    ifdet_groups: Dict[Tuple[DALY_vid, int], List[int]] = {}
    for ifdet, fdet in enumerate(fdets):
        vf_id = (fdet['ind'][0], fdet['ind'][1])
        ifdet_groups.setdefault(vf_id, []).append(ifdet)
    ifgt_to_ifdets: Dict[int, List[int]] = {}
    for ifgt, fgt in enumerate(fgts):
        vf_id = (fgt['ind'][0], fgt['ind'][1])
        ifgt_to_ifdets[ifgt] = ifdet_groups.get(vf_id, [])
    ifdet_to_ifgts: Dict[int, List[int]] = {}
    for ifgt, ifdets in ifgt_to_ifdets.items():
        for ifdet in ifdets:
            ifdet_to_ifgts.setdefault(ifdet, []).append(ifgt)
    # Preparation
    detection_matched = np.zeros(len(fdets), dtype=bool)
    gt_already_matched = np.zeros(len(fgts), dtype=bool)
    # Provenance
    detection_matched_to_which_gt = np.ones(len(fdets), dtype=int)*-1
    iou_coverages_per_detection_ind: Dict[int, List[float]] = {}

    # VOC2007 preparation
    nd = len(fdets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if use_diff:
        npos = len(fgts)
    else:
        npos = len([x for x in fgts if not x['diff']])

    # Go through ordered detections
    detection_scores = np.array([x['score'] for x in fdets])
    detection_scores = detection_scores.round(3)
    sorted_inds = np.argsort(-detection_scores)
    for d, ifdet in enumerate(sorted_inds):
        # Check available GTs
        share_image_ifgts: List[int] = ifdet_to_ifgts.get(ifdet, [])
        if not len(share_image_ifgts):
            fp[d] = 1
            continue

        detection: AP_fdet_framebox = fdets[ifdet]
        detection_box = detection['obj']

        # Compute IOUs
        iou_coverages: List[float] = []
        for ifgt in share_image_ifgts:
            gt_box_anno: AP_fgt_framebox = fgts[ifgt]
            gt_box = gt_box_anno['obj']
            iou = numpy_iou(gt_box, detection_box)
            iou_coverages.append(iou)
        # Provenance
        iou_coverages_per_detection_ind[ifdet] = iou_coverages

        max_coverage_local_id = np.argmax(iou_coverages)
        max_coverage = iou_coverages[max_coverage_local_id]
        max_coverage_ifgt = share_image_ifgts[max_coverage_local_id]

        # Mirroring voc_eval
        if max_coverage > iou_thresh:
            if (not use_diff) and fgts[max_coverage_ifgt]['diff']:
                continue
            if not gt_already_matched[max_coverage_ifgt]:
                tp[d] = 1
                detection_matched[ifdet] = True
                gt_already_matched[max_coverage_ifgt] = True
                detection_matched_to_which_gt[ifdet] = max_coverage_ifgt
            else:
                fp[d] = 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap


def voclike_legacy_evaluation_v2(
        object_names: List[str],
        datalist,
        predicted_datalist):
    # // Transform to sensible data format
    o_fgts: Dict[str, List[AP_fgt_framebox]] = \
            {on: [] for on in object_names}
    for record in datalist:
        vid = record['vid']
        iframe = record['video_frame_number']
        for anno_id, anno in enumerate(record['annotations']):
            object_name = object_names[anno['category_id']]
            fgt: AP_fgt_framebox = {
                'ind': (vid, iframe, anno_id),
                'obj': anno['bbox'],
                'diff': False
            }
            o_fgts[object_name].append(fgt)
    # # We assume vid+frame is enough to assign  det to gt
    # o_gt_to_det: Dict[str,
    #         Dict[Tuple[DALY_vid, int], List[int]]] = {}
    o_fdets: Dict[str, List[AP_fdet_framebox]] = \
            {on: [] for on in object_names}
    for record, pred_item in zip(datalist, predicted_datalist):
        vid = record['vid']
        iframe = record['video_frame_number']
        pred_boxes = pred_item.pred_boxes.tensor.numpy()
        scores = pred_item.scores.numpy()
        pred_classes = pred_item.pred_classes.numpy()
        for det_id, (bbox, score, cls_ind) in enumerate(
                zip(pred_boxes, scores, pred_classes)):
            object_name = object_names[cls_ind]
            fdet: AP_fdet_framebox = {
                'ind': (vid, iframe, anno_id),
                'obj': bbox,
                'score': score
            }
            o_fdets[object_name].append(fdet)
    # Params
    use_07_metric = False
    use_diff = False
    iou_thresh = 0.5
    object_classes = object_names
    ap_per_cls: Dict[str, float] = {}
    for obj_cls in object_classes:
        ap = evaluate_voc_detections_v2(
            o_fgts[obj_cls],
            o_fdets[obj_cls],
            iou_thresh, use_07_metric, use_diff)
        ap_per_cls[obj_cls] = ap

    # Results printed nicely via pd.Series
    x = pd.Series(ap_per_cls)*100
    x.loc['AVERAGE'] = x.mean()
    table = snippets.string_table(
            np.array(x.reset_index()),
            header=['Object', 'AP'],
            col_formats=['{}', '{:.2f}'], pad=2)
    log.info(f'AP@{iou_thresh:.3f}:\n{table}')
