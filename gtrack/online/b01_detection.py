# -*- coding: utf-8 -*-
"""
① 물체 검출 (YOLO)

경계상자 중심을 영상 대표점으로 사용한다.
출력(raw 엑셀 이름): Top_Status / Front_Status, Top_U_px / Top_V_px / Front_U_px / Front_V_px,
                    Top_Conf / Front_Conf
코드 내부 이름      : Status_Top / Status_Front, Top_px_u ..., Top_conf / Front_conf

- resolve_class_id : 모델의 클래스 이름 목록에서 대상 클래스 번호를 찾는다(TARGET_CLASS_NAMES).
- pick_best_target : 대상 클래스 중 conf 최대 상자의 중심 픽셀과 conf를 반환한다.
"""

import numpy as np

from .. import settings


def pick_best_target(res, target_cls):
    """★ 반환: (중심픽셀 [x,y] 또는 None, conf)"""
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0: return None, np.nan
    mask = (boxes.cls == target_cls)
    if mask.sum() == 0: return None, np.nan
    conf, xywh = boxes.conf[mask], boxes.xywh[mask]
    i = conf.argmax()
    return xywh[i].tolist()[:2], float(conf[i])


def resolve_class_id(model, wanted=None):
    """[2026-09-30] 모델의 클래스 이름 목록에서 대상 클래스 번호를 하나만 찾는다."""
    if wanted is None:
        wanted = settings.TARGET_CLASS_NAMES
    norm = lambda s: str(s).lower().replace(' ', '').replace('_', '')
    names = {int(k): norm(v) for k, v in model.names.items()}
    wanted_norm = {norm(w) for w in wanted}
    hits = [k for k, v in names.items() if v in wanted_norm]
    if len(hits) != 1:
        raise ValueError(f'대상 클래스 확인 필요: 찾는 이름 {wanted}, 모델 클래스 {model.names}')
    return hits[0]


def load_models():
    """상단·정면 YOLO 가중치를 읽고 대상 클래스 번호를 찾는다."""
    from ultralytics import YOLO
    model_top, model_front = YOLO(settings.TOP_MODEL_PATH), YOLO(settings.FRONT_MODEL_PATH)
    cls_top = resolve_class_id(model_top)
    cls_front = resolve_class_id(model_front)
    print(f'Target class: top={cls_top} {model_top.names[cls_top]!r}, '
          f'front={cls_front} {model_front.names[cls_front]!r}')
    return model_top, model_front, cls_top, cls_front


def select_yolo_device():
    """GPU가 있으면 0, 없으면 'cpu'."""
    try:
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    print(f"YOLO device: {device}")
    return device


class CachedDetector:
    """한 카메라의 YOLO 검출기.

    낮은 FPS 영상의 같은 원본 프레임이 재사용되면 YOLO 결과도 재사용한다.
    detect() 반환: (YOLO 결과, 중심 픽셀 [u, v] 또는 None, conf)
    """

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
    """다이어그램 마름모 "두 시점 검출 성립?".

    성립하면 None을 반환하고 ②③으로 진행한다.
    성립하지 않으면 Missing_Reason(TOP / FRONT / BOTH_MISSING 등)을 반환하고 ④로 바로 간다.
    """
    if not sync_ok:
        return 'SYNC_OUT_OF_TOLERANCE'
    elif pt_top is None and pt_front is None:
        return 'BOTH_MISSING'
    elif pt_top is None:
        return 'TOP_MISSING'
    elif pt_front is None:
        return 'FRONT_MISSING'
    return None
