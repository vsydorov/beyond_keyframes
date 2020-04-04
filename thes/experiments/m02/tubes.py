import copy
import itertools
import pandas as pd
import warnings
import logging
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import (List, Tuple, Dict, cast, TypedDict, Set, Sequence)
from types import MethodType

import torch
from torch.nn import functional as F

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as d2_transforms
from detectron2.structures import Boxes, Instances

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data.dataset.external import (Dataset_daly_ocv, Vid_daly)
from thes.caffe import (Nicolas_net_helper)
from thes.detectron.cfg import (
    set_detectron_cfg_base, set_detectron_cfg_test,)
from thes.detectron.externals import (simple_d2_setup,)
from thes.detectron.daly import (
    get_daly_split_vids, simplest_daly_to_datalist_v2,
    get_datalist_action_object_converter,)
from thes.data.tubes.types import (
    DALY_wein_tube, DALY_wein_tube_index, Objaction_dets, Frametube,
    Sframetube, convert_dwein_tube, convert_dgt_tubes, dtindex_filter_split,
    av_filter_split, av_stubes_above_score, get_daly_gt_tubes, AV_dict)
from thes.data.tubes.routines import (
    filter_tube_keyframes_only_gt_v2,
    compute_nms_for_av_stubes, score_ftubes_via_objaction_overlap_aggregation,)
from thes.evaluation.routines import (
    compute_recall_for_avtubes_as_dfs, compute_ap_for_avtubes_as_df)
from thes.tools import snippets


log = logging.getLogger(__name__)


class Box_connections_dwti(TypedDict):
    vid: Vid_daly
    frame_ind: int
    dwti_sources: List[DALY_wein_tube_index]  # N
    boxes: List[np.ndarray]  # N, 4


def compute_recall_ap_for_avtubes_as_dfdict(
        av_gt_tubes: AV_dict[Frametube],
        av_stubes: AV_dict[Sframetube],
        iou_thresholds: List[float]
        ) -> Dict[str, pd.DataFrame]:
    df_recall_s, df_recall_st = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, av_stubes, iou_thresholds)
    df_ap_s = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, False)
    df_ap_st = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, True)
    dfdict = {'recall_s': df_recall_s, 'recall_st': df_recall_st,
            'ap_s': df_ap_s, 'ap_st': df_ap_st}
    return dfdict

def computeprint_recall_ap_for_avtubes(
        av_gt_tubes: AV_dict[Frametube],
        av_stubes: AV_dict[Sframetube],
        iou_thresholds: List[float]):
    """
    Will compute tube ap per threshold, print table per thresh,
    print aggregate table
    """
    # Get in DF form
    dfdict = compute_recall_ap_for_avtubes_as_dfdict(
            av_gt_tubes, av_stubes, iou_thresholds)
    # Convert to str_tables
    tables = {k: snippets.df_to_table_v2((v*100).round(2))
            for k, v in dfdict.items()}
    # Print
    log.info('Spatial Recall:\n{}'.format(tables['recall_s']))
    log.info('Spatiotemp Recall:\n{}'.format(tables['recall_st']))
    log.info('Spatial AP:\n{}'.format(tables['ap_s']))
    log.info('Spatiotemp AP:\n{}'.format(tables['ap_st']))


class Ncfg_dataset:
    @staticmethod
    def set_dataset_seed(cfg):
        cfg.set_deftype("""
        dataset:
            name: [~, ['daly']]
            cache_folder: [~, str]
            subset: ['test', str]
        seed: [42, int]
        """)

    @staticmethod
    def resolve_dataset_tubes(cf):
        dataset = Dataset_daly_ocv()
        dataset.populate_from_folder(cf['dataset.cache_folder'])
        split_label = cf['dataset.subset']
        split_vids = get_daly_split_vids(dataset, split_label)
        av_gt_tubes: AV_dict[Frametube] = \
                convert_dgt_tubes(get_daly_gt_tubes(dataset))
        av_gt_tubes = av_filter_split(av_gt_tubes, split_vids)
        return dataset, split_vids, av_gt_tubes


class Ncfg_tubes:
    @staticmethod
    def set_defcfg(cfg):
        """
        wein.leave_only_gt_keyframes:
            only keyframes that overlap with gt keyframes are left
        """
        cfg.set_deftype("""
        tubes:
            source: ['wein', ['wein', 'gt']]
            wein:
                path: [~, ~]
                leave_only_gt_keyframes:
                    enabled: [False, bool]
                    keep_temporal: [True, bool]
        """)

    @staticmethod
    def resolve_tubes(
            cf,
            av_gt_tubes: AV_dict[Frametube],
            split_vids: List[Vid_daly]
            ) -> Dict[DALY_wein_tube_index, Frametube]:
        ftubes: Dict[DALY_wein_tube_index, Frametube]
        if cf['tubes.source'] == 'wein':
            ftubes = resolve_wein_tubes(
                    av_gt_tubes, split_vids, cf['tubes.wein.path'],
                    cf['tubes.wein.leave_only_gt_keyframes.enabled'],
                    cf['tubes.wein.leave_only_gt_keyframes.keep_temporal'])
        elif cf['tubes.source'] == 'gt':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return ftubes


class Ncfg_nicphil_rcnn:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_defaults("""
        rcnn:
            PIXEL_MEANS: [102.9801, 115.9465, 122.7717]
            TEST_SCALES: [600,]
            TEST_MAX_SIZE: 1000
        """)

    @staticmethod
    def resolve_helper(cf):
        neth = Nicolas_net_helper(cf['rcnn.PIXEL_MEANS'],
                cf['rcnn.TEST_SCALES'], cf['rcnn.TEST_MAX_SIZE'])
        return neth


class Ncfg_tube_eval:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        tube_eval:
            enabled: [True, bool]
            minscore_cutoff: [0.05, float]
            nms:
                enabled: [True, bool]
                thresh: [0.5, float]
            iou_thresholds: [[0.3, 0.5, 0.7], list]
        """)

    @staticmethod
    def evalprint_if(cf,
            av_stubes: AV_dict[Sframetube],
            av_gt_tubes: AV_dict[Frametube]):
        if not cf['tube_eval.enabled']:
            return
        av_stubes = av_stubes_above_score(
                av_stubes, cf['tube_eval.minscore_cutoff'])
        if cf['tube_eval.nms.enabled']:
            av_stubes = compute_nms_for_av_stubes(
                    av_stubes, cf['tube_eval.nms.thresh'])
        computeprint_recall_ap_for_avtubes(
                av_gt_tubes, av_stubes, cf['tube_eval.iou_thresholds'])

    @staticmethod
    def eval_as_df(cf,
            av_stubes: AV_dict[Sframetube],
            av_gt_tubes: AV_dict[Frametube]):
        assert cf['tube_eval.enabled']
        av_stubes = av_stubes_above_score(
                av_stubes, cf['tube_eval.minscore_cutoff'])
        if cf['tube_eval.nms.enabled']:
            av_stubes = compute_nms_for_av_stubes(
                    av_stubes, cf['tube_eval.nms.thresh'])
        dfdict = compute_recall_ap_for_avtubes_as_dfdict(
                av_gt_tubes, av_stubes, cf['tube_eval.iou_thresholds'])
        return dfdict


def _set_rcnn_vid_eval_defcfg(cfg):
    cfg.set_deftype("""
    demo_run:
        enabled: [False, bool]
        N: [50, int]
        seed: [0, int]
    compute:
        save_period: ['::10', str]
        split:
            enabled: [False, bool]
            chunk: [0, "VALUE >= 0"]
            total: [1, int]
            equal: ['frames', ['frames', 'tubes']]
    """)


def resolve_wein_tubes(
        av_gt_tubes: AV_dict[Frametube],
        split_vids: List[Vid_daly],
        wein_path: Path,
        l_enabled: bool,
        l_keeptemp: bool):
    dwein_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
            small.load_pkl(wein_path)
    dwein_tubes = dtindex_filter_split(dwein_tubes, split_vids)
    # Convert dwein_tubes to sparse tubes
    ftubes = {k: convert_dwein_tube(t) for k, t in dwein_tubes.items()}
    # Filter tubes optionally
    if l_enabled:
        ftubes = filter_tube_keyframes_only_gt_v2(
                ftubes, av_gt_tubes, l_keeptemp)
    return ftubes


def _resolve_actobjects(cf, dataset, split_vids):
    # / Assign objects to tubes
    # // Create objaction_dets in video frames
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = {}
    datalist = _recreate_actobject_datalist(dataset, split_vids)
    if cf['actobjects.source'] == 'detected':
        # /// Load detections themselves
        actobjects_evaluated = small.load_pkl(cf['actobjects.detected.path'])
        # /// Assign objactions
        for dl_item, pred_item in zip(datalist, actobjects_evaluated):
            pred_boxes = pred_item.pred_boxes.tensor.numpy()
            scores = pred_item.scores.numpy()
            pred_classes = pred_item.pred_classes.numpy()
            pred_classes = np.array([dataset.action_names[i]
                for i in pred_classes])
            detections: Objaction_dets = {
                    'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes}
            (objactions_vf
                .setdefault(dl_item['vid'], {})
                [dl_item['video_frame_number']]) = detections
    elif cf['actobjects.source'] == 'gt':
        # /// Create fake "perfect" detections
        for dl_item in datalist:
            pred_boxes = []
            pred_classes = []
            for anno in dl_item['annotations']:
                pred_boxes.append(anno['bbox'])
                pred_classes.append(
                        dataset.action_names[anno['category_id']])
            pred_boxes = np.array(pred_boxes)
            pred_classes = np.array(pred_classes)
            scores = np.ones(len(pred_boxes))
            detections: Objaction_dets = {
                    'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes}
            (objactions_vf
                .setdefault(dl_item['vid'], {})
                [dl_item['video_frame_number']]) = detections
    else:
        raise NotImplementedError()
    return objactions_vf


def sample_dict(dct: Dict, N=10, NP_SEED=0) -> Dict:
    np_rstate = np.random.RandomState(NP_SEED)
    prm_key_indices = np_rstate.permutation(np.arange(len(dct)))
    key_list = list(dct.keys())
    some_keys = [key_list[i] for i in prm_key_indices[:N]]
    some_tubes = {k: dct[k] for k in some_keys}
    return some_tubes


def _recreate_actobject_datalist(dataset, split_vids):
    # /// Recreate the datalist that was used for detections
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)
    object_names, datalist_converter = \
            get_datalist_action_object_converter(dataset)
    datalist = datalist_converter(datalist)
    return datalist


def equal_tube_split(tubes_per_video, ct, split_kind):
    key_indices = np.arange(len(tubes_per_video))
    key_list = list(tubes_per_video.keys())

    # Simple tube df
    nframes_df = []
    for k, v in tubes_per_video.items():
        vid = k[0]
        nframes = len(v['frame_inds'])
        nframes_df.append([vid, nframes])
    nframes_df = pd.DataFrame(nframes_df, columns=['vid', 'nframes'])
    nframes_df['keys'] = key_list

    # Divide indices
    if split_kind == 'tubes':
        equal_split = np.array_split(key_indices, ct)
    elif split_kind == 'frames':
        approx_nframes_per_split = nframes_df.nframes.sum() // ct
        approx_split_indices = approx_nframes_per_split * np.arange(1, ct)
        split_indices = np.searchsorted(
                nframes_df.nframes.cumsum(), approx_split_indices)
        equal_split = np.array_split(key_indices, split_indices)
    else:
        raise NotImplementedError()

    # Assign splits
    for i, inds in enumerate(equal_split):
        nframes_df.loc[inds, 'split'] = i
    nframes_df['split'] = nframes_df['split'].astype(int)

    # Compute stats
    gb_chunk = nframes_df.groupby('split')
    all_nvids = gb_chunk['vid'].unique().apply(len)
    all_nframes = gb_chunk['nframes'].sum()
    split_stats = pd.concat((all_nvids, all_nframes), axis=1)

    # Divide tubes
    split_tubes = [{} for i in range(ct)]
    for i, group in gb_chunk.groups.items():
        keys = nframes_df.loc[group, 'keys'].tolist()
        for k in keys:
            split_tubes[i][k] = tubes_per_video[k]
    return split_tubes, split_stats


def _parcel_management(cf, tubes_per_video):
    # // Computation of parcels
    cc, ct = (cf['compute.chunk'], cf['compute.total'])
    split_kind = cf['compute.equal_split']
    split_tubes, split_stats = \
            equal_tube_split(tubes_per_video, ct, split_kind)
    ctubes_per_video = split_tubes[cc]
    # Logging part
    log.info('Chunk {}/{}: {} -> {}'.format(
        cc, ct, len(tubes_per_video), len(ctubes_per_video)))
    log.info('split_stats:\n{}'.format(split_stats))
    return ctubes_per_video


def get_daly_keyframes(
        dataset: Dataset_daly_ocv, split_vids) -> Dict[Vid_daly, np.ndarray]:
    to_cover_: Dict[Vid_daly, Set] = {}
    for vid in split_vids:
        v = dataset.videos_ocv[vid]
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                frames = [kf['frame'] for kf in instance['keyframes']]
                to_cover_[vid] = to_cover_.get(vid, set()) | set(list(frames))
    frames_to_cover = \
            {k: np.array(sorted(v)) for k, v in to_cover_.items()}
    return frames_to_cover


def prepare_ftube_box_computations(
        ftubes: Dict[DALY_wein_tube_index, Frametube],
        frames_to_cover: Dict[Vid_daly, np.ndarray]
        ) -> Dict[Vid_daly, Dict[int, Box_connections_dwti]]:
    # Assign boxes (and keep connections to original ftubes)
    vf_connections_dwti_list: Dict[Vid_daly, Dict[int,
        List[Tuple[DALY_wein_tube_index, np.ndarray]]]] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        tube_finds = tube['frame_inds']
        good_finds = frames_to_cover[vid]
        common_finds, comm1, comm2 = np.intersect1d(
            tube_finds, good_finds, assume_unique=True, return_indices=True)
        if len(common_finds) == 0:
            continue
        good_tube_boxes = tube['boxes'][comm1]
        for find, box in zip(common_finds, good_tube_boxes):
            (vf_connections_dwti_list
                .setdefault(vid, {})
                .setdefault(find, []).append((dwt_index, box)))
    # Prettify
    vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = {}
    for vid, f_connections_dwti_list in vf_connections_dwti_list.items():
        for find, connections_dwti_list in f_connections_dwti_list.items():
            lsources, lboxes = zip(*connections_dwti_list)
            boxes = np.vstack(lboxes)
            bcs: Box_connections_dwti = {
                'vid': vid,
                'frame_ind': find,
                'dwti_sources': lsources,
                'boxes': boxes
            }
            vf_connections_dwti.setdefault(vid, {})[find] = bcs
    return vf_connections_dwti


def _demovis_apply_pncaffe_rcnn(
        neth, dataset: Dataset_daly_ocv, vf_connections_dwti, out):
    vfold = small.mkdir(out/'demovis')
    nicolas_labels = ['background', ] + cast(List[str], dataset.action_names)
    for vid, f_connections_dwti in tqdm(
            vf_connections_dwti.items(), 'nicphil_demovis'):
        video_path = dataset.videos[vid]['path']
        finds = list(f_connections_dwti)
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, finds, debug_filename=video_path)

        video_fold = small.mkdir(vfold/f'vid{vid}')

        for find, frame_BGR in zip(finds, frames_u8):
            connections_dwti = f_connections_dwti[find]
            boxes = connections_dwti['boxes']
            box_cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
            # Draw and print
            txt_output = []
            image = frame_BGR.copy()
            for i, cls_probs in enumerate(box_cls_probs):
                box = boxes[i]
                best_score_id = np.argmax(cls_probs)
                best_score = cls_probs[best_score_id]
                best_nicolas_label = nicolas_labels[best_score_id]
                snippets.cv_put_box_with_text(image, box,
                    text='{} {} {:.2f}'.format(
                        i, best_nicolas_label, best_score))
                line = (' '.join([f'{y}: {x:.3f}'
                    for x, y in zip(cls_probs, nicolas_labels)])
                    + str(box))
                txt_output.append(line)
            cv2.imwrite(str(
                video_fold/'Fr{:05d}.png'.format(find)), image)
            with (video_fold/f'Fr{find:05d}_scores.txt').open('w') as f:
                f.write('\n'.join(txt_output))


def _predict_rcnn_given_box_resized_proposals(
        box4, frame_u8, transform_gen, model):

    o_height, o_width = frame_u8.shape[:2]
    got_transform = transform_gen.get_transform(frame_u8)

    # Transform image
    image = got_transform.apply_image(frame_u8)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    imshape = tuple(image.shape[1:3])

    # / Transform box
    assert box4.shape == (4,)
    boxes_unscaled = box4[None]
    t_boxes = torch.as_tensor(boxes_unscaled.astype("float32"))
    transformed_t_boxes = got_transform.apply_box(t_boxes)
    # // Proposals w.r.t transformed imagesize
    proposal = Instances(imshape)
    tb_boxes = Boxes(transformed_t_boxes)
    proposal.proposal_boxes = tb_boxes

    inputs = {
            "image": image,
            "proposals": proposal,
            "height": o_height,
            "width": o_width}

    with torch.no_grad():
        predictions = model([inputs])[0]
    return predictions


def genrcnn_rcnn_roiscores_forward(self, batched_inputs):
    """
    Replacing detectron2/detectron2/modeling/meta_arch/rcnn.py (GeneralizedRCNN.forward)
    """
    assert not self.training
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    assert "proposals" in batched_inputs[0]
    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
    del images
    # Borrowed from detectron2/detectron2/modeling/roi_heads/roi_heads.py (Res5ROIHeads.forward)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = self.roi_heads._shared_roi_transform(
        [features[f] for f in self.roi_heads.in_features], proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
    pred_class_logits, pred_proposal_deltas = \
            self.roi_heads.box_predictor(feature_pooled)
    pred_softmax = F.softmax(pred_class_logits, dim=-1)
    return pred_softmax


class D2_rcnn_helper(object):
    def __init__(self, cf, cf_add_d2, dataset, out):
        num_classes = len(dataset.action_names)
        TEST_DATASET_NAME = 'daly_objaction_test'

        # / Define d2 conf
        d2_output_dir = str(small.mkdir(out/'d2_output'))
        d_cfg = set_detectron_cfg_base(
                d2_output_dir, num_classes, cf['seed'])
        d_cfg = set_detectron_cfg_test(
                d_cfg, TEST_DATASET_NAME,
                cf['d2_rcnn.model'], cf['d2_rcnn.conf_thresh'], cf_add_d2,
                freeze=False)
        d_cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
        d_cfg.freeze()

        # / Start d2
        simple_d2_setup(d_cfg)

        # Predictor without proposal generator
        model = build_model(d_cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)

        checkpointer.load(d_cfg.MODEL.WEIGHTS)
        MIN_SIZE_TEST = d_cfg.INPUT.MIN_SIZE_TEST
        MAX_SIZE_TEST = d_cfg.INPUT.MAX_SIZE_TEST
        transform_gen = d2_transforms.ResizeShortestEdge(
            [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)

        # Instance monkeypatching
        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module/50600307#50600307
        model.forward = MethodType(genrcnn_rcnn_roiscores_forward, model)

        self.d_cfg = d_cfg
        self.rcnn_roiscores_model = model
        self.cpu_device = torch.device("cpu")
        self.transform_gen = transform_gen

    def score_boxes(self, frame_BGR, boxes) -> np.ndarray:
        o_height, o_width = frame_BGR.shape[:2]
        got_transform = self.transform_gen.get_transform(frame_BGR)
        # Transform image
        image = got_transform.apply_image(frame_BGR)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        imshape = tuple(image.shape[1:3])
        # Transform box
        t_boxes = torch.as_tensor(boxes.astype("float32"))
        transformed_t_boxes = got_transform.apply_box(t_boxes)
        # Proposals w.r.t transformed imagesize
        proposal = Instances(imshape)
        tb_boxes = Boxes(transformed_t_boxes)
        proposal.proposal_boxes = tb_boxes
        inputs = {
                "image": image,
                "proposals": proposal,
                "height": o_height,
                "width": o_width}
        with torch.no_grad():
            pred_softmax = self.rcnn_roiscores_model([inputs])
        X = pred_softmax.to(self.cpu_device).numpy()
        # To conform to caffe style put background cls at 0th position
        X_caffelike = np.c_[X[:, -1:], X[:, :-1]]
        return X_caffelike


def _rcnn_vid_eval(
        cf, out, dataset: Dataset_daly_ocv,
        split_vids, ftubes, neth):
    # Cover only keyframes when evaluating dwti tubes
    frames_to_cover = get_daly_keyframes(dataset, split_vids)
    vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = \
            prepare_ftube_box_computations(ftubes, frames_to_cover)

    if cf['demo_run.enabled']:
        vf_connections_dwti = sample_dict(
            vf_connections_dwti, N=5, NP_SEED=0)
        _demovis_apply_pncaffe_rcnn(
                neth, dataset, vf_connections_dwti, out)
        return

    def isaver_eval_func(vid):
        f_connections_dwti = vf_connections_dwti[vid]
        video_path = dataset.videos[vid]['path']
        finds = list(f_connections_dwti)
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, finds, debug_filename=video_path)
        f_cls_probs = {}
        for find, frame_BGR in zip(finds, frames_u8):
            connections_dwti = f_connections_dwti[find]
            boxes = connections_dwti['boxes']
            cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
            f_cls_probs[find] = cls_probs
        return f_cls_probs

    assert not cf['compute.split.enabled']

    isaver_keys = list(vf_connections_dwti.keys())
    isaver = snippets.Simple_isaver(
            small.mkdir(out/'isave_rcnn_vid_eval'),
            isaver_keys, isaver_eval_func,
            cf['compute.save_period'], 120)

    isaver_items = isaver.run()
    vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
    vf_cls_probs = dict(zip(isaver_keys, isaver_items))

    # Pretty clear we'll be summing up the scores anyway
    ftube_scores = {k: np.zeros(11) for k in ftubes}
    for vid, f_cls_probs in vf_cls_probs.items():
        for f, cls_probs in f_cls_probs.items():
            dwtis = vf_connections_dwti[vid][f]['dwti_sources']
            for dwti, prob in zip(dwtis, cls_probs):
                ftube_scores[dwti] += prob

    # Create av_stubes
    av_stubes: AV_dict[Sframetube] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(
                dataset.action_names, ftube_scores[dwt_index][1:]):
            stube = tube.copy()
            stube['score'] = score
            stube = cast(Sframetube, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes


# Experiments


def assign_objactions_to_tubes(workfolder, cfg_dict, add_args):
    """
    Score tubes by assigning objactions to them and pooling the scores,
    then evaluate resulting scored tubes
    - Objactions: detecton evaluated datalist or gt objects (per frame)
    - Tubes: philippe tubes
    - Assignment: inner overlap or iou scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    cfg.set_deftype("""
    actobjects:
        source: ['detected', ['detected', 'gt']]
        detected:
            path: [~, ~]

    obj_to_tube:
        overlap_type: ['inner_overlap', ['inner_overlap', 'iou']]
        overlap_cutoff: [0.2, float]
        score_cutoff: [0.2, float]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = Ncfg_dataset.resolve_dataset_tubes(cf)
    # Inputs to the assignment routine
    ftubes: Dict[DALY_wein_tube_index, Frametube] = \
            Ncfg_tubes.resolve_tubes(cf, av_gt_tubes, split_vids)
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = \
            _resolve_actobjects(cf, dataset, split_vids)
    # Assignment itself
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    av_stubes: AV_dict[Sframetube] = \
        score_ftubes_via_objaction_overlap_aggregation(dataset,
        objactions_vf, ftubes, overlap_type, overlap_cutoff, score_cutoff)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pncaffe_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Apply Phil-Nic rcnn model on tube boxes to extract per-action scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    Ncfg_nicphil_rcnn.set_defcfg(cfg)
    _set_rcnn_vid_eval_defcfg(cfg)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = Ncfg_dataset.resolve_dataset_tubes(cf)
    ftubes: Dict[DALY_wein_tube_index, Frametube] = \
            Ncfg_tubes.resolve_tubes(cf, av_gt_tubes, split_vids)
    neth: Nicolas_net_helper = Ncfg_nicphil_rcnn.resolve_helper(cf)

    av_stubes = _rcnn_vid_eval(cf, out, dataset, split_vids, ftubes, neth)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pfadet_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Apply trained d2 frcnn model on tube boxes to extract per-action scores
      - We dispense with the frcnn box predictions and only use per-roi scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    _set_rcnn_vid_eval_defcfg(cfg)
    cfg.set_deftype("""
    d2_rcnn:
        model: [~, ~]
        conf_thresh: [0.0, float]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset, split_vids, av_gt_tubes = Ncfg_dataset.resolve_dataset_tubes(cf)
    ftubes: Dict[DALY_wein_tube_index, Frametube] = \
            Ncfg_tubes.resolve_tubes(cf, av_gt_tubes, split_vids)
    neth = D2_rcnn_helper(cf, cf_add_d2, dataset, out)

    av_stubes = _rcnn_vid_eval(cf, out, dataset, split_vids, ftubes, neth)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def _meanpool_avstubes(abstubes_to_merge: Sequence[AV_dict[Sframetube]]):
    av_stubes: AV_dict[Sframetube] = {}
    for a, v_dict in abstubes_to_merge[0].items():
        for vid, stubes in v_dict.items():
            for i, stube in enumerate(stubes):
                scores = [t[a][vid][i]['score'] for t in abstubes_to_merge]
                new_stube = stube.copy()
                new_stube['score'] = np.mean(scores)
                (av_stubes
                        .setdefault(a, {})
                        .setdefault(vid, []).append(new_stube))
    return av_stubes


def merge_scores_avstubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tube_eval.set_defcfg(cfg)
    cfg.set_defaults("""
    tube_dict: ~
    combinations:
        enabled: False
        sizes: ~

    """)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = Ncfg_dataset.resolve_dataset_tubes(cf)
    ts = {k: small.load_pkl(v) for k, v in cfg_dict['tube_dict'].items()}
    if not cf['combinations.enabled']:
        av_stubes = _meanpool_avstubes(list(ts.values()))
        small.save_pkl(out/'merged_av_stubes.pkl', av_stubes)
        log.info('All combined score:')
        Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)
        return

    sizes = cf['combinations.sizes']
    combinations = [list(itertools.combinations(ts.keys(), r)) for r in sizes]
    combinations = list(itertools.chain(*combinations))
    log.info('Combinations: {}'.format(combinations))

    comb_dfdicts = {}
    for comb in combinations:
        comb_name = '+'.join(comb)
        comb_fold = small.mkdir(out/comb_name)

        def compute():
            to_merge = [ts[k] for k in comb]
            av_stubes = _meanpool_avstubes(to_merge)
            small.save_pkl(comb_fold/'av_stubes.pkl', av_stubes)
            dfdict = Ncfg_tube_eval.eval_as_df(cf, av_stubes, av_gt_tubes)
            return dfdict

        dfdict = small.stash2(comb_fold/'stashed_dfdict.pkl')(compute)
        comb_dfdicts[comb_name] = dfdict

    log.info('Individual results:')
    for comb, dfdict in comb_dfdicts.items():
        log.info(f'Results for {comb_name}:')
        tables = {k: snippets.df_to_table_v2((v*100).round(2))
                for k, v in dfdict.items()}
        log.info('Spatial Recall:\n{}'.format(tables['recall_s']))
        log.info('Spatiotemp Recall:\n{}'.format(tables['recall_st']))
        log.info('Spatial AP:\n{}'.format(tables['ap_s']))
        log.info('Spatiotemp AP:\n{}'.format(tables['ap_st']))

    log.info('Combined tables:')
    big_= {comb: pd.concat(dfdict)
            for comb, dfdict in comb_dfdicts.items()}
    big = pd.concat(big_, axis=1)
    for stat in big.index.levels[0]:
        log.info(f'=== {stat} ===')
        for thresh in big.columns.levels[1]:
            X = (big.loc['ap_st']
                .loc[:, pd.IndexSlice[:, thresh]]
                .droplevel(1, axis=1))
            table = snippets.df_to_table_v2((X*100).round(2))
            log.info(f'{stat} for IOU {thresh}:\n{table}')
        log.info('\n')
