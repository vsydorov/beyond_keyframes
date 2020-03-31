import logging
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
from typing import (  # NOQA
    Dict, List, Tuple, TypeVar, Set, Optional, Callable, TypedDict, NewType,
    NamedTuple, Sequence, Literal, cast)

from thes.tools import snippets
from thes.data.dataset.external import (
        DatasetDALY, DALY_vid,
        DALY_action_name)

from thes.data.tubes.types import (
        Sframetube, Frametube, Base_frametube,
        DALY_wein_tube_index, DALY_gt_tube_index,
        Objaction_dets,
        DALY_gt_tube,
        get_daly_gt_tubes, V_dict, AV_dict)

from vsydorov_tools import small

log = logging.getLogger(__name__)


T = TypeVar('T')


def gt_tubes_to_df(
        dataset: DatasetDALY,
        gt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube]
            ) -> pd.DataFrame:
    gt_df_list = []
    for dgt_index, v in gt_tubes.items():
        vid, action_name, ins_ind = dgt_index
        vmp4 = dataset.source_videos[vid]
        ocv_video_fps = vmp4['frames_reached']/vmp4['length_reached']
        # vmeta = dataset.video_odict[k[0]]
        # meta_video_fps = vmeta['fps']
        # if ocv_video_fps != meta_video_fps:
        #     log.info('FPS mismatch at {}: OCV: {} META: {}'.format(
        #         k, ocv_video_fps, meta_video_fps))
        min_kframe = min(v['frame_inds'])
        max_kframe = max(v['frame_inds'])
        start_frame = int(v['start_time']*ocv_video_fps)
        end_frame = int(v['end_time']*ocv_video_fps)
        frame_inds = v['frame_inds']
        n_frames = len(frame_inds)
        gt_df_list.append([
            vid, action_name, ins_ind,
            min_kframe, max_kframe,
            start_frame, end_frame, frame_inds, n_frames])
    gt_df = pd.DataFrame(gt_df_list)
    gt_df.columns = ['vid', 'action', 'ins_id',
            'min_kframe', 'max_kframe',
            'start_frame', 'end_frame',
            'frame_inds', 'n_frames']
    return gt_df


def filter_tube_keyframes_only_gt(
        dataset: DatasetDALY,
        tubes: Dict[DALY_wein_tube_index, Frametube]
            ) -> Dict[DALY_wein_tube_index, Frametube]:
    dgt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube] = \
            get_daly_gt_tubes(dataset)
    gt_df: pd.DataFrame = gt_tubes_to_df(dataset, dgt_tubes)
    # Query good inds per vid
    good_inds_per_vid: Dict[DALY_vid, List[int]] = {}
    for vid, gindices in gt_df.groupby('vid').groups.items():
        qdf = gt_df.loc[gindices]
        sorted_inds = sorted(
                itertools.chain.from_iterable(qdf.frame_inds.tolist()))
        good_inds_per_vid[vid] = sorted_inds
    # Filter tubes to only gt keyframes
    ftubes: Dict[DALY_wein_tube_index, Frametube] = {}
    for dwt_index, v in tqdm(tubes.items(), 'filter_tubes'):
        (vid, bunch_id, tube_id) = dwt_index
        good_inds = good_inds_per_vid[vid]
        intersecting_inds, comm1, comm2 = \
            np.intersect1d(v['frame_inds'], good_inds, return_indices=True)
        if len(intersecting_inds):
            frame_inds = v['frame_inds'][comm1]
            boxes = v['boxes'][comm1]
            v_intersect: Frametube = {
                'frame_inds': frame_inds,
                'boxes': boxes,
                'start_frame': v['start_frame'],
                'end_frame': v['end_frame']}
            ftubes[dwt_index] = v_intersect
    return ftubes


def filter_tube_keyframes_only_gt_v2(
        tubes: Dict[DALY_wein_tube_index, Frametube],
        av_gttubes: AV_dict[Frametube],
        keep_temporal: bool,
            ) -> Dict[DALY_wein_tube_index, Frametube]:
    """
    Filter "tubes" to contain only those frames,
    which are present in the DALY GT annotations
    """

    # Query good inds per vid
    gtinds: Dict[DALY_vid, Set[int]] = {}
    for action_name, v_gttubes in av_gttubes.items():
        for vid, gttubes in v_gttubes.items():
            for t in gttubes:
                gtinds[vid] = gtinds.get(vid, set()) | set(t['frame_inds'])

    # Filter tubes to only gt keyframes
    ftubes: Dict[DALY_wein_tube_index, Frametube] = {}
    for dwt_index, v in tqdm(tubes.items(), 'filter_tubes'):
        (vid, bunch_id, tube_id) = dwt_index
        good_inds: List[int] = list(gtinds.get(vid, set()))
        intersecting_inds, comm1, comm2 = \
            np.intersect1d(v['frame_inds'], good_inds, return_indices=True)
        if len(intersecting_inds):
            frame_inds = v['frame_inds'][comm1]
            boxes = v['boxes'][comm1]
            if keep_temporal:
                start_frame = v['start_frame']
                end_frame = v['end_frame']
            else:
                start_frame = np.min(frame_inds)
                end_frame = np.max(frame_inds)
            v_intersect: Frametube = {
                'frame_inds': frame_inds,
                'boxes': boxes,
                'start_frame': start_frame,
                'end_frame': end_frame}
            ftubes[dwt_index] = v_intersect
    return ftubes


def ex_tubes_to_df(extracted_tubes):
    ex_df = []
    for k, v in extracted_tubes.items():
        min_frame = v['frame_inds'].min()
        max_frame = v['frame_inds'].max()
        ex_df.append([*k, min_frame, max_frame])
    ex_df = pd.DataFrame(ex_df)
    ex_df.columns = ['vid', 'bunch_id', 'tube_id', 'min_frame', 'max_frame']
    return ex_df


def temporal_coverage_stats(ex_df, gt_df):
    # Let's compute temporal coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        if len(interesting_frange):
            # Clip within range, compute "intersect" part
            clipped_frange = np.clip(interesting_frange, s, e)
            total_gt = e - s
            total_intersect = clipped_frange[:, 1] - clipped_frange[:, 0]
            # Compute union
            union_frange = np.empty_like(interesting_frange)
            union_frange[:, 0] = np.minimum(interesting_frange[:, 0], s)
            union_frange[:, 1] = np.maximum(interesting_frange[:, 1], e)
            total_union = union_frange[:, 1] - union_frange[:, 0]
            # Compute fraction
            fraction_intersect = total_intersect/total_gt
            fraction_iou = total_intersect/total_union
            max_intersect = np.max(fraction_intersect)
            max_iou = np.max(fraction_iou)
        else:
            max_intersect = 0.0
            max_iou = 0.0
        coverage_dict[key] = [max_intersect, max_iou]

    coverage_df = pd.DataFrame(coverage_dict).T
    coverage_df.columns = ['minter', 'miou']

    # Compute stats
    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.miou.mean() * 100
    stats['mean_inter'] = cdf.minter.mean() * 100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.miou >= 0.5).sum()
    stats['tubes_above_iou_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_minter05 = (cdf.minter >= 0.5).sum()
    stats['tubes_above_inter_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_minter05, N, N_tubes_above_minter05/N * 100)
    return coverage_df, stats


def spatial_coverage_stats(ex_df, gt_df, dataset, extracted_tubes):
    # Let's compute spatial coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        tubes_inside = vid_tubes.iloc[either_limit_inside]
        if len(interesting_frange):
            # // Compute keyframe intersections
            # Retrieve GT keyframes
            gt_instance = (dataset.video_odict[line.vid]
                    ['instances'][line.action][line.ins_id])
            vmp4 = dataset.source_videos[line.vid]
            gt_frames = []
            gt_boxes_unscaled = []
            for kf in gt_instance['keyframes']:
                gt_frames.append(kf['frameNumber'])
                gt_boxes_unscaled.append(kf['boundingBox'])
            gt_frames = np.array(gt_frames)
            gt_boxes_unscaled = np.vstack(gt_boxes_unscaled)
            gt_boxes = gt_boxes_unscaled * np.tile(
                    [vmp4['width'], vmp4['height']], 2)
            # Retrieve those keyframes from proposals that match gt_frames
            retrieved = []
            for i, tube_row in tubes_inside.iterrows():
                ext_tube = extracted_tubes[
                        tube_row.vid, tube_row.bunch_id, tube_row.tube_id]
                found = np.isin(gt_frames, ext_tube['frame_inds'])
                found_ind = np.searchsorted(ext_tube['frame_inds'], gt_frames)
                found_ind[~found] = 0
                found_boxes = ext_tube['boxes'][found_ind]
                retrieved.append({'boxes': found_boxes, 'found': found})
            # Compute pairwise box IOUs
            pairwise_box_ious = []
            for i, x in enumerate(retrieved):
                ious = []
                for gt_box, p_box, found in zip(
                        gt_boxes, x['boxes'], x['found']):
                    if not found:
                        iou = 0.0
                    else:
                        # Computing IOU
                        inter = np.r_[
                            np.maximum(gt_box[:2], p_box[:2]),
                            np.minimum(gt_box[2:], p_box[2:])]
                        if np.any(inter[:2] > inter[2:]):
                            iou = 0.0
                        else:
                            inter_area = np.prod(inter[2:] - inter[:2])
                            union_area = (
                                np.prod(gt_box[2:] - gt_box[:2]) +
                                np.prod(p_box[2:] - p_box[:2]) - inter_area)
                            iou = inter_area/union_area
                    ious.append(iou)
                pairwise_box_ious.append(ious)
            pairwise_box_ious = np.array(pairwise_box_ious)
            # Mean per GT frame
            mean_box_ious = np.mean(pairwise_box_ious, axis=1)
            # Maximum iou
            max_iou = np.max(mean_box_ious)
        else:
            max_iou = 0.0
        coverage_dict[key] = max_iou

    coverage_df = pd.Series(coverage_dict).to_frame()
    coverage_df.columns = ['max_iou']

    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.max_iou.mean()*100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.max_iou > 0.5).sum()
    stats['N_tubes_above_iou05'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_iou03 = (cdf.max_iou > 0.3).sum()
    stats['N_tubes_above_iou03'] = '{}/{} = {}'.format(
            N_tubes_above_iou03, N, N_tubes_above_iou03/N * 100)

    # Stats per action
    acdf = cdf[['action', 'max_iou']].copy()
    acdf['iou_above_05'] = acdf['max_iou'] > 0.5
    acdf['iou_above_03'] = acdf['max_iou'] > 0.3
    sum_per_action = acdf.groupby('action').sum()
    count_per_action = acdf.groupby('action').count()
    stats_df_peraction = sum_per_action/count_per_action*100
    return coverage_df, stats, stats_df_peraction


def stats_of_wein_tubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    extracted_tubes = small.load_pkl(cf['imported_wein_tubes'])
    gt_tubes = get_daly_gt_tubes(dataset)

    # We reference video only in terms of ocv frames from now on
    # We ASSUME extracted philippe frames here are the OCV frames
    # Probably this is not the case
    ex_df = ex_tubes_to_df(extracted_tubes)
    gt_df = gt_tubes_to_df(dataset, gt_tubes)

    # coverage_df, temp_stats = temporal_coverage_stats(ex_df, gt_df)
    coverage_df, spat_stats = spatial_coverage_stats(
            ex_df, gt_df, dataset, extracted_tubes)


def nicphil_evaluations_to_tubes(
        dataset, tubes_per_video, original_tubes_per_video, tubescores_dict):
    stubes_va: \
            Dict[DALY_action_name,
                    Dict[DALY_vid, List[Sframetube]]] = {}
    for ckey, tube in tubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        original_tube = original_tubes_per_video[ckey]
        tubescores = tubescores_dict[ckey]
        agg_tubescores = np.vstack(tubescores).sum(0)[1:]
        start_frame = original_tube['frame_inds'].min()
        end_frame = original_tube['frame_inds'].max()
        # Sum the perframe scores
        for action_name, score in zip(
                dataset.action_names, agg_tubescores):
            sparse_scored_tube = {
                    'frame_inds': tube['frame_inds'],
                    'boxes': tube['boxes'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'score': score}
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sparse_scored_tube))
    return stubes_va


def _barea(box):
    return np.prod(box[2:] - box[:2])


def _bareas(boxes):
    return np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)


def _inter_areas(boxes1, boxes2):
    inter = np.c_[
        np.maximum(boxes1[..., :2], boxes2[..., :2]),
        np.minimum(boxes1[..., 2:], boxes2[..., 2:])]
    inter_subs = inter[..., 2:] - inter[..., :2]
    inter_areas = np.prod(inter_subs, axis=1)
    inter_areas[(inter_subs < 0).any(axis=1)] = 0.0
    return inter_areas


def numpy_iou_11(box1, box2):
    assert box1.shape == (4,)
    assert box2.shape == (4,)
    # Computing IOU
    inter = np.r_[
        np.maximum(box1[:2], box2[:2]),
        np.minimum(box1[2:], box2[2:])]
    if np.any(inter[:2] >= inter[2:]):
        iou = 0.0
    else:
        inter_area = _barea(inter)
        box1_area = _barea(box1)
        box2_area = _barea(box2)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area/union_area
    return iou


def numpy_inner_overlap_N1(boxes1, box2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    assert box2.shape == (4,)
    inter_areas = _inter_areas(boxes1, box2)
    boxes1_areas = _bareas(boxes1)
    ioverlaps = inter_areas / boxes1_areas
    return ioverlaps


def numpy_iou_N1(boxes1, box2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    assert box2.shape == (4,)
    inter_areas = _inter_areas(boxes1, box2)
    boxes1_areas = _bareas(boxes1)
    box2_area = _barea(box2)
    union_areas = boxes1_areas + box2_area - inter_areas
    ious = inter_areas / union_areas
    return ious


def numpy_iou_NN(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    inter_areas = _inter_areas(boxes1, boxes2)
    boxes1_areas = _bareas(boxes1)
    boxes2_areas = _bareas(boxes2)
    union_areas = boxes1_areas + boxes2_areas - inter_areas
    ious = inter_areas / union_areas
    return ious


def nms_over_custom_elements(
        element_list: List[T],
        overlaps_func: Callable[[T, Sequence[T]], List[float]],
        score_func: Callable[[T], float],
        thresh: float,
        ) -> List[T]:
    scores = [score_func(e) for e in element_list]
    sorted_ids = np.argsort(scores)[::-1]  # In decreasing order
    sorted_candidates = [element_list[i] for i in sorted_ids]
    results = []
    while len(sorted_candidates):
        taken = sorted_candidates.pop(0)
        results.append(taken)
        overlaps = overlaps_func(taken, sorted_candidates)
        sorted_candidates = [
                c for c, o in zip(sorted_candidates, overlaps) if o < thresh]
    return results


def temporal_IOU(
        b1, e1, b2, e2):
    begin = max(b1, b2)
    end = min(e1, e2)
    inter = end-begin+1
    if inter <= 0:
        return 0.0
    else:
        union = (e1 - b1 + 1) + (e2 - b2 + 1) - inter
        return inter/union


def spatial_tube_iou_v3(
        tube1: Base_frametube,
        tube2: Base_frametube,
        ) -> float:
    """
    Compute avg iou over matching keyframes
    """
    ii, c1, c2 = np.intersect1d(
            tube1['frame_inds'], tube2['frame_inds'],
            assume_unique=True, return_indices=True)
    if len(ii):
        c1_boxes = tube1['boxes'][c1]
        c2_boxes = tube2['boxes'][c2]
        ious = numpy_iou_NN(c1_boxes, c2_boxes)
        miou = np.mean(ious)
    else:
        miou = np.nan
    return miou


def temporal_ious_where_positive(x_bf, x_ef, y_frange):
    """
    Temporal ious between inter X and multiple Y inters
    Inputs:
        x_bg, x_ef - temporal range of input
    Returns 2 np.ndarrays:
        pids: indices of ytubes with >0 temporal iou
        ptious: >0 temporal ious
    """
    if len(y_frange) == 0:
        pids = np.array([], dtype=np.int)
        ptious = np.array([])
        return ptious, pids
    ibegin = np.maximum(y_frange[:, 0], x_bf)
    iend = np.minimum(y_frange[:, 1], x_ef)
    temporal_intersections = iend-ibegin+1
    pids = np.where(temporal_intersections > 0)[0]
    if len(pids) == 0:
        ptious = np.array([])
    else:
        ptemp_inters = temporal_intersections[pids]
        p_bfs, p_efs = y_frange[pids].T
        ptemp_unions = (x_ef - x_bf + 1) + (p_efs - p_bfs + 1) - ptemp_inters
        ptious = ptemp_inters/ptemp_unions
    return ptious, pids


def spatiotemp_tube_iou_1N(
        x: Sframetube, ys: Sequence[Sframetube]) -> np.ndarray:
    """
    Spatiotemporal IOUs: x tube with every y tube
    """
    y_frange = np.array([(y['start_frame'], y['end_frame']) for y in ys])
    ptious, pids = temporal_ious_where_positive(
            x['start_frame'], x['end_frame'], y_frange)
    st_overlaps = np.zeros(len(ys))
    if len(pids):
        pys = [ys[pid] for pid in pids]
        pmious = [spatial_tube_iou_v3(y, x) for y in pys]
        st_overlaps[pids] = ptious * pmious
    return st_overlaps


def compute_nms_for_v_stubes(
        v_stubes: V_dict[Sframetube],
        thresh: float) -> V_dict[Sframetube]:
    v_stubes_nms = {}
    for vid, tubes in tqdm(v_stubes.items(), desc='nms'):
        nmsed_tubes = nms_over_custom_elements(
            tubes, spatiotemp_tube_iou_1N, lambda x: x['score'], thresh)
        v_stubes_nms[vid] = nmsed_tubes
    return v_stubes_nms


def computecache_nms_for_av_stubes(
        av_stubes: AV_dict[Sframetube],
        thresh: float,
        nms_folder) -> AV_dict[Sframetube]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        nmsed_stubes_v = small.stash2(
            nms_folder/f'scored_tubes_nms_{thresh:.2f}_at_{a}_v2.pkl')(
            compute_nms_for_v_stubes,
            v_stubes, thresh)
        av_stubes_nms[a] = nmsed_stubes_v
    return av_stubes_nms


def compute_nms_for_av_stubes(
        av_stubes: AV_dict[Sframetube],
        thresh: float,
        ) -> AV_dict[Sframetube]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        av_stubes_nms[a] = compute_nms_for_v_stubes(v_stubes, thresh)
    return av_stubes_nms


def score_ftubes_via_objaction_overlap_aggregation(
        objactions_vf: Dict[DALY_vid, Dict[int, Objaction_dets]],
        ftubes: Dict[DALY_wein_tube_index, Frametube],
        overlap_type: Literal['inner_overlap', 'iou'],
        overlap_cutoff: float,
        score_cutoff: float
        ) -> AV_dict[Sframetube]:
    """
    """
    # To every tube, find matching keyframes
    dwti_ascore: Dict[DALY_wein_tube_index, Dict[DALY_action_name, float]] = {}
    for dwt_index, tube in tqdm(ftubes.items(), 'match_keyframes'):
        (vid, bunch_id, tube_id) = dwt_index
        cls_scores: Dict[DALY_action_name, float] = {}
        for frame_ind, tube_box in zip(
                tube['frame_inds'], tube['boxes']):
            # In frame, match box to all objections
            odets: Optional[Objaction_dets] = \
                    objactions_vf.get(vid, {}).get(frame_ind)
            if odets is None:
                continue
            # Check score
            score_above = odets['scores'] > score_cutoff
            sa_boxes = odets['pred_boxes'][score_above]
            # Check overlap
            if overlap_type == 'iou':
                sa_overlaps = numpy_iou_N1(sa_boxes, tube_box)
            elif overlap_type == 'inner_overlap':
                sa_overlaps = numpy_inner_overlap_N1(sa_boxes, tube_box)
            else:
                raise RuntimeError()
            sa_overlap_above = sa_overlaps > overlap_cutoff
            sa_oa_scores = odets['scores'][score_above][sa_overlap_above]
            sa_oa_classes = odets['pred_classes'][score_above][sa_overlap_above]
            for score, cls in zip(sa_oa_scores, sa_oa_classes):
                cls_scores[cls] = cls_scores.get(cls, 0.0) + score
        dwti_ascore[dwt_index] = cls_scores
    # Score the ftubes, convert to av_dict
    av_stubes: AV_dict[Sframetube] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        scores: Dict[DALY_action_name, float] = dwti_ascore[dwt_index]
        # Sum the perframe scores
        for action_name, score in scores.items():
            stube = tube.copy()
            stube['score'] = score
            stube = cast(Sframetube, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes
