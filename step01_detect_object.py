"""Step ①: YOLO object detection; the box center is the image point of each camera."""

import numpy as np

from .. import settings


def pick_best_target(res, target_cls):
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0: return None, np.nan
    mask = (boxes.cls == target_cls)
    if mask.sum() == 0: return None, np.nan
    conf, xywh = boxes.conf[mask], boxes.xywh[mask]
    i = conf.argmax()
    return xywh[i].tolist()[:2], float(conf[i])


def resolve_class_id(model, wanted=None):
    if wanted is None:
        wanted = settings.TARGET_CLASS_NAMES
    norm = lambda s: str(s).lower().replace(' ', '').replace('_', '')
    names = {int(k): norm(v) for k, v in model.names.items()}
    wanted_norm = {norm(w) for w in wanted}
    hits = [k for k, v in names.items() if v in wanted_norm]
    if len(hits) != 1:
        raise ValueError(f'Target class not found: wanted {wanted}, model classes {model.names}')
    return hits[0]


def load_models():
    from ultralytics import YOLO
    model_top, model_front = YOLO(settings.TOP_MODEL_PATH), YOLO(settings.FRONT_MODEL_PATH)
    cls_top = resolve_class_id(model_top)
    cls_front = resolve_class_id(model_front)
    print(f'Target class: top={cls_top} {model_top.names[cls_top]!r}, '
          f'front={cls_front} {model_front.names[cls_front]!r}')
    return model_top, model_front, cls_top, cls_front


def select_yolo_device():
    try:
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    print(f"YOLO device: {device}")
    return device


class CachedDetector:
    def __init__(self, model, class_id, predict_kwargs):
        self.model = model
        self.class_id = class_id
        self.predict_kwargs = predict_kwargs
        self.cache_source_frame = None
        self.cache_res = None
        self.cache_pt = None
        self.cache_conf = np.nan

    def detect(self, frame, source_frame):
        if source_frame == self.cache_source_frame:
            return self.cache_res, self.cache_pt, self.cache_conf
        res = self.model.predict(frame, **self.predict_kwargs)
        pt, conf = pick_best_target(res, self.class_id)
        self.cache_source_frame = source_frame
        self.cache_res = res
        self.cache_pt = pt
        self.cache_conf = conf
        return res, pt, conf


def pairing_reason(sync_ok, pt_top, pt_front):
    if not sync_ok:
        return 'SYNC_OUT_OF_TOLERANCE'
    elif pt_top is None and pt_front is None:
        return 'BOTH_MISSING'
    elif pt_top is None:
        return 'TOP_MISSING'
    elif pt_front is None:
        return 'FRONT_MISSING'
    return None
