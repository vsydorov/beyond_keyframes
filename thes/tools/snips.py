import logging
from typing import (Dict, List, Tuple, cast, NewType,
        Any, TypedDict, Optional, Literal)

import cv2

from thes.tools import video as tvideo

log = logging.getLogger(__name__)


# Smaller snippets


class OCV_rstats(TypedDict):
    # OCV reachability stats
    height: int
    width: int
    frame_count: int
    fps: float
    max_pos_frames: int  # 1-based
    max_pos_msec: float


def compute_ocv_rstats(video_path, n_tries=5) -> OCV_rstats:
    with tvideo.video_capture_open(video_path, n_tries) as vcap:
        height, width = tvideo.video_getHW(vcap)
        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vcap.get(cv2.CAP_PROP_FPS)
        while True:
            max_pos_frames = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
            max_pos_msec = vcap.get(cv2.CAP_PROP_POS_MSEC)
            ret = vcap.grab()
            if ret is False:
                break
    ocv_rstats: OCV_rstats = {
        'height': height,
        'width': width,
        'frame_count': frame_count,
        'fps': fps,
        'max_pos_frames': max_pos_frames,
        'max_pos_msec': max_pos_msec,
        }
    return ocv_rstats
