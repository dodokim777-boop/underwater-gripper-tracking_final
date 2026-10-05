# -*- coding: utf-8 -*-
"""
② 광선 생성 · 평면 경계 굴절

구현은 gtrack/geometry.py에 있다 (후처리 ⑨⑩도 같은 함수를 사용하므로 공용 모듈로 둠).
벡터형 Snell 굴절을 적용하며, 굴절 ON · OFF 후보는 ③에서 같은 픽셀로 동시에 계산한다.
"""

from ..geometry import _camera_ray_world, _refract_at_plane  # noqa: F401


__all__ = ['_camera_ray_world', '_refract_at_plane']
