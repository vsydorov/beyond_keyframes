"""
Combines "daly.py" and "external.py" into a more reasonable structure
"""
import platform
import copy
import xml.etree.ElementTree as ET
import subprocess
import csv
import hashlib
import re
import pandas as pd
import numpy as np
import cv2
import logging
import concurrent.futures
from abc import abstractmethod, ABC
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from typing import (Dict, List, Tuple, cast, NewType,
        Any, TypedDict, Optional, Literal)

from vsydorov_tools import small, cv as vt_cv

from thes.filesystem import get_dataset_path
from thes.tools import snippets
from thes.tools.snips import (OCV_rstats, compute_ocv_rstats)


log = logging.getLogger(__name__)


class Instance_flags_daly(TypedDict):
    isSmall: bool
    isReflection: bool
    isShotcut: bool
    isZoom: bool
    isAmbiguous: bool
    isOccluded: bool
    isOutsideFOV: bool


class Keyframe_daly(TypedDict):
    # shape (1, 4), LTRD[xmin, ymin, xmax, ymax], relative (0..1)
    boundingBox: np.ndarray
    # [xmin, ymin, xmax, ymax, objectID, isOccluded, isHallucinate]
    objects: np.ndarray
    frameNumber: int  # 1-based
    pose: np.ndarray
    time: float  # seconds


class Instance_daly(TypedDict):
    beginTime: float
    endTime: float
    flags: Instance_flags_daly
    keyframes: List[Keyframe_daly]


class Video_daly(TypedDict):
    vid: str
    path: Path
    suggestedClass: str
    instances: Dict[str, List[Instance_daly]]
    # Provided meta
    duration: float
    nbframes_ffmpeg: int
    fps: float


class Dataset_daly(object):
    root_path: Path
    action_names = [
        'ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor',
        'CleaningWindows', 'Drinking', 'FoldingTextile', 'Ironing',
        'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos']
    object_names = [
        'balm', 'bedsheet', 'bottle', 'bowl', 'broom', 'brush',
        'camera', 'cloth', 'cup', 'electricToothbrush', 'finger',
        'glass', 'glass+straw', 'gourd', 'hand', 'harmonica', 'hat',
        'iron', 'mobilePhone', 'mop', 'moppingMachine', 'newspaper',
        'other', 'pencil', 'phone', 'plasticBag', 'q-tip',
        'satellitePhone', 'scrubber', 'shirt', 'smartphone',
        'soap', 'sponge', 'spoon', 'squeegee', 'steamCleaner',
        'stick', 't-shirt', 'toothbrush', 'towel', 'trousers',
        'vase', 'videocamera']
    joint_names = [
        'head', 'shoulderLeft', 'elbowLeft', 'wristLeft',
        'shoulderRight', 'elbowRight', 'wristRight']
    split: Dict[Vid_daly, Dataset_subset]
    provided_metas: Dict[Vid_daly, ProvidedMetadata_daly]
    videos: Dict[Vid_daly, Video_daly]

    def __init__(self, mirror):
        super().__init__()
        self.root_path = get_dataset_path('action/daly_take2')
        if mirror == 'uname':
            GOOD_NODES = ['gpuhost6', 'gpuhost7', 'gpuhost9', 'services']
            fallback_mirror = 'scratch2'
            node_name = platform.uname().node
            if node_name in GOOD_NODES:
                mirror = node_name
            else:
                mirror = fallback_mirror
            log.info(f'{mirror=} set according to uname')
        self.mirror = mirror
        self._load_pkl()

    def _load_pkl(self):
        pkl_path = self.root_path/'annotations/daly1.1.0.pkl'
        info = small.load_py2_pkl(pkl_path)
        assert self.action_names == info['labels']
        assert self.object_names == info['objectList']
        assert self.joint_names == info['joints']
        videos: Dict[Vid_daly, Video_daly] = {}
        provided_metas: Dict[Vid_daly, ProvidedMetadata_daly] = {}
        for video_name, v in info['annot'].items():
            vid = video_name.split('.')[0]
            video_meta = info['metadata'][video_name]
            video: Video_daly = {
                'vid': vid,
                'path': (self.root_path/
                    f'mirrors/{self.mirror}/{vid}.mp4'),
                'suggestedClass': v['suggestedClass'],
                'instances': v['annot']}
            meta: ProvidedMetadata_daly = {
                'duration': video_meta['duration'],
                'nbframes_ffmpeg': video_meta['nbframes_ffmpeg'],
                'fps': video_meta['fps']}
            videos[vid] = video
            provided_metas[vid] = meta
        split = {k: 'train' for k in videos.keys()}
        for video_name in info['splits'][0]:
            vid = video_name.split('.')[0]
            split[vid] = 'test'
        self.videos = videos
        self.provided_metas = provided_metas
        self.split = split
