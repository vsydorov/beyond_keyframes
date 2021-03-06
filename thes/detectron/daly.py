"""
Detectron interactions with the DALY dataset
"""
import numpy as np
import pandas as pd
import copy
from pathlib import Path
from typing import (  # NOQA
    List, Dict, Tuple, cast, TypedDict, Callable,
    Optional, Literal, Union)

from detectron2.structures import BoxMode  # type: ignore

from thes.data.dataset.external import (
        Dataset_daly, Dataset_daly_ocv, Vid_daly,
        Action_name_daly, Object_name_daly)


class Dl_anno(TypedDict):
    bbox: np.ndarray
    bbox_mode: BoxMode
    category_id: int
    is_occluded: bool


class Dl_record(TypedDict):
    vid: Vid_daly
    video_path: Path
    video_frame_number: int
    video_frame_time: float
    action_name: Action_name_daly
    image_id: str
    height: int
    width: int
    annotations: List[Dl_anno]


Datalist = List[Dl_record]


def daly_to_datalist_pfadet(
        dataset: Dataset_daly_ocv, split_vids) -> Datalist:
    d2_datalist = []
    for vid in split_vids:
        ovideo = dataset.videos_ocv[vid]
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                for keyframe in instance['keyframes']:
                    frame0 = keyframe['frame']
                    frame_time = keyframe['time']
                    image_id = '{}_A{}_FN{}_FT{:.3f}'.format(
                            vid, action_name, frame0, frame_time)
                    action_id = dataset.action_names.index(action_name)
                    act_obj: Dl_anno = {
                            'bbox': keyframe['bbox_abs'],
                            'bbox_mode': BoxMode.XYXY_ABS,
                            'category_id': action_id,
                            'is_occluded': False}
                    annotations = [act_obj]
                    record: Dl_record = {
                            'vid': vid,
                            'video_path': ovideo['path'],
                            'video_frame_number': frame0,
                            'video_frame_time': frame_time,
                            'action_name': action_name,
                            'image_id': image_id,
                            'height': ovideo['height'],
                            'width': ovideo['width'],
                            'annotations': annotations}
                    d2_datalist.append(record)
    return d2_datalist


def simplest_daly_to_datalist_v2(
        dataset: Dataset_daly_ocv, split_vids) -> Datalist:
    d2_datalist = []
    for vid in split_vids:
        ovideo = dataset.videos_ocv[vid]
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                for keyframe in instance['keyframes']:
                    frame0 = keyframe['frame']
                    frame_time = keyframe['time']
                    image_id = '{}_A{}_FN{}_FT{:.3f}'.format(
                            vid, action_name, frame0, frame_time)
                    kf_objects_abs = keyframe['objects_abs']
                    annotations = []
                    for kfo in kf_objects_abs:
                        [xmin, ymin, xmax, ymax,
                            objectID, isOccluded, isHallucinate] = kfo
                        isOccluded = bool(isOccluded)
                        isHallucinate = bool(isHallucinate)
                        if isHallucinate:
                            continue
                        bbox = np.array([xmin, ymin, xmax, ymax])
                        obj: Dl_anno = {
                                'bbox': bbox,
                                'bbox_mode': BoxMode.XYXY_ABS,
                                'category_id': int(objectID),
                                'is_occluded': isOccluded}
                        annotations.append(obj)
                    if len(annotations) == 0:
                        continue
                    record: Dl_record = {
                            'vid': vid,
                            'video_path': ovideo['path'],
                            'video_frame_number': frame0,
                            'video_frame_time': frame_time,
                            'action_name': action_name,
                            'image_id': image_id,
                            'height': ovideo['height'],
                            'width': ovideo['width'],
                            'annotations': annotations}
                    d2_datalist.append(record)
    return d2_datalist


def get_daly_odf(dataset: Dataset_daly_ocv):
    gt_objects = []
    for vid, ovideo in dataset.videos_ocv.items():
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                for keyframe in instance['keyframes']:
                    kf_objects = keyframe['objects']
                    frame_number = keyframe['frameNumber']
                    for kfo in kf_objects:
                        [xmin, ymin, xmax, ymax,
                            objectID, isOccluded, isHallucinate] = kfo
                        bbox = np.array([xmin, ymin, xmax, ymax])
                        objectID = int(objectID)
                        object_name = dataset.object_names[objectID]
                        obj = {
                                'vid': vid,
                                'ins_ind': ins_ind,
                                'frame_number': frame_number,
                                'action_name': action_name,
                                'bbox': bbox,
                                'object_name': object_name,
                                'is_occluded': isOccluded,
                                'is_hallucinate': isHallucinate}
                        gt_objects.append(obj)
    odf = pd.DataFrame(gt_objects)
    return odf


def get_category_map_o100(dataset: Dataset_daly_ocv):
    # o100 computations
    odf = get_daly_odf(dataset)
    ocounts = odf.object_name.value_counts()
    o100_objects = sorted(ocounts[ocounts>100].index)
    category_map = []
    for obj_name in dataset.object_names:
        mapped: Union[int, None]
        if obj_name in o100_objects:
            mapped = o100_objects.index(obj_name)
        else:
            mapped = None
        category_map.append(mapped)
    return o100_objects, category_map


def make_datalist_o100(d2_datalist, category_map):
    filtered_datalist = []
    for record in d2_datalist:
        filtered_annotations = []
        for obj in record['annotations']:
            new_category_id = category_map[obj['category_id']]
            if new_category_id is not None:
                new_obj = copy.copy(obj)
                new_obj['category_id'] = new_category_id
                filtered_annotations.append(new_obj)
        if len(filtered_annotations) != 0:
            new_record = copy.copy(record)
            new_record['annotations'] = filtered_annotations
            filtered_datalist.append(new_record)
    return filtered_datalist


def make_datalist_objaction_similar_merged(
        d2_datalist: Datalist,
        old_object_names: List[str],
        new_object_names: List[str],
        action_object_to_object: Dict) -> Datalist:

    filtered_datalist = []
    for record in d2_datalist:
        filtered_annotations = []
        action_name = record['action_name']
        for obj in record['annotations']:
            old_object_name = old_object_names[obj['category_id']]
            new_object_name = action_object_to_object[
                    (action_name, old_object_name)]
            if new_object_name is None:
                continue
            new_category_id = new_object_names.index(new_object_name)
            new_obj = copy.copy(obj)
            new_obj['category_id'] = new_category_id
            filtered_annotations.append(new_obj)
        if len(filtered_annotations) != 0:
            new_record = copy.copy(record)
            new_record['annotations'] = filtered_annotations
            filtered_datalist.append(new_record)
    return filtered_datalist


def get_similar_action_objects_DALY() -> Dict[Tuple[Action_name_daly, Object_name_daly], str]:
    """ Group similar looking objects, ignore other ones """
    action_object_to_object = \
        {('ApplyingMakeUpOnLips', 'balm'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'brush'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'finger'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'pencil'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'q-tip'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'stick'): 'ApplyingMakeUpOnLips_stick_like',
         ('BrushingTeeth', 'electricToothbrush'): 'BrushingTeeth_toothbrush_like',
         ('BrushingTeeth', 'toothbrush'): 'BrushingTeeth_toothbrush_like',
         ('CleaningFloor', 'broom'): 'CleaningFloor_mop_like',
         ('CleaningFloor', 'brush'): None,
         ('CleaningFloor', 'cloth'): None,
         ('CleaningFloor', 'mop'): 'CleaningFloor_mop_like',
         ('CleaningFloor', 'moppingMachine'): None,
         ('CleaningFloor', 'steamCleaner'): None,
         ('CleaningWindows', 'cloth'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'newspaper'): None,
         ('CleaningWindows', 'scrubber'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'soap'): None,
         ('CleaningWindows', 'sponge'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'squeegee'): 'CleaningWindows_squeegee_like',
         ('Drinking', 'bottle'): None,
         ('Drinking', 'bowl'): None,
         ('Drinking', 'cup'): 'Drinking_glass_like',
         ('Drinking', 'glass'): 'Drinking_glass_like',
         ('Drinking', 'glass+straw'): 'Drinking_glass_like',
         ('Drinking', 'gourd'): None,
         ('Drinking', 'hand'): None,
         ('Drinking', 'hat'): None,
         ('Drinking', 'other'): None,
         ('Drinking', 'plasticBag'): None,
         ('Drinking', 'spoon'): None,
         ('Drinking', 'vase'): None,
         ('FoldingTextile', 'bedsheet'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'cloth'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'shirt'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 't-shirt'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'towel'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'trousers'): 'FoldingTextile_bedsheet_like',
         ('Ironing', 'iron'): 'Ironing_iron_like',
         ('Phoning', 'mobilePhone'): 'Phoning_phone_like',
         ('Phoning', 'phone'): 'Phoning_phone_like',
         ('Phoning', 'satellitePhone'): 'Phoning_phone_like',
         ('Phoning', 'smartphone'): 'Phoning_phone_like',
         ('PlayingHarmonica', 'harmonica'): 'PlayingHarmonica_harmonica_like',
         ('TakingPhotosOrVideos', 'camera'): 'TakingPhotosOrVideos_camera_like',
         ('TakingPhotosOrVideos', 'smartphone'): 'TakingPhotosOrVideos_camera_like',
         ('TakingPhotosOrVideos', 'videocamera'): 'TakingPhotosOrVideos_camera_like'}
    return cast(Dict[Tuple[Action_name_daly, Object_name_daly], str], action_object_to_object)


def get_datalist_action_object_converter(
        dataset: Dataset_daly,
        ) -> Tuple[List[str], Callable[[Datalist], Datalist]]:
    action_object_to_object = get_similar_action_objects_DALY()
    object_names = sorted([x
        for x in set(list(action_object_to_object.values())) if x])

    def datalist_converter(datalist):
        datalist = make_datalist_objaction_similar_merged(
                datalist, dataset.object_names, object_names,
                action_object_to_object)
        return datalist

    return object_names, datalist_converter


def get_biggest_objects_DALY() -> Tuple[Action_name_daly, Object_name_daly]:
    """ Biggest object category per action class """
    primal_configurations = [
            ('ApplyingMakeUpOnLips', 'stick'),
            ('BrushingTeeth', 'toothbrush'),
            ('CleaningFloor', 'mop'),
            ('CleaningWindows', 'squeegee'),
            ('Drinking', 'glass'),
            ('FoldingTextile', 'bedsheet'),
            ('Ironing', 'iron'),
            ('Phoning', 'phone'),
            ('PlayingHarmonica', 'harmonica'),
            ('TakingPhotosOrVideos', 'camera')]
    return cast(Tuple[Action_name_daly, Object_name_daly], primal_configurations)
