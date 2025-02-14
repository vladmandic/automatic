# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
from PIL import Image
from installer import installed, pip, log
from modules.control.util import HWC3, resize_image
from .draw import draw_bodypose, draw_handpose, draw_facepose
checked_ok = False


def check_dependencies():
    global checked_ok # pylint: disable=global-statement
    debug = log.trace if os.environ.get('SD_DWPOSE_DEBUG', None) is not None else lambda *args, **kwargs: None
    packages = [
        'openmim==0.3.9',
        'mmengine==0.10.4',
        'mmcv==2.1.0',
        'mmpose==1.3.1',
        'mmdet==3.3.0',
    ]
    status = [installed(p, reload=False, quiet=False) for p in packages]
    debug(f'DWPose required={packages} status={status}')
    if not all(status):
        log.info(f'Installing DWPose dependencies: {[packages]}')
        cmd = 'install --upgrade --no-deps --force-reinstall '
        pkgs = ' '.join(packages)
        res = pip(cmd + pkgs, ignore=False, quiet=False)
        debug(f'DWPose pip install: {res}')
    try:
        import pkg_resources
        import imp # pylint: disable=deprecated-module
        imp.reload(pkg_resources)
        import mmcv # pylint: disable=unused-import
        import mmengine # pylint: disable=unused-import
        import mmpose # pylint: disable=unused-import
        import mmdet # pylint: disable=unused-import
        debug('DWPose import ok')
        checked_ok = True
    except Exception as e:
        log.error(f'DWPose: {e}')
    return checked_ok


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands)
    canvas = draw_facepose(canvas, faces)
    return canvas


class DWposeDetector:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu"):
        self.pose_estimation = None
        if not checked_ok:
            if not check_dependencies():
                return
        from .wholebody import Wholebody
        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, device)

    def to(self, device):
        self.pose_estimation.to(device)
        return self

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", min_confidence=0.3, **kwargs):
        if self.pose_estimation is None:
            log.error("DWPose: not loaded")
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, _C = input_image.shape

        candidate, subset = self.pose_estimation(input_image)
        if candidate is None:
            return Image.fromarray(input_image)
        nums, _keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]

        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > min_confidence:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1
        un_visible = subset < min_confidence
        candidate[un_visible] = -1
        _foot = candidate[:,18:24]
        faces = candidate[:,24:92]
        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)
        detected_map = draw_pose(pose, H, W)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
