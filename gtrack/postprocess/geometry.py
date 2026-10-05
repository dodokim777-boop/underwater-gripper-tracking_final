# -*- coding: utf-8 -*-
"""
후처리 공용 기하 함수

- top_ray_plane_z    : 상단 굴절 광선 × Z 평면 교점 (TOP_RAY 복원에서 X, Y 계산)
- front_ray_plane_y  : 정면 굴절 광선 × Y 평면 교점 (FRONT_RAY 복원에서 X, Z 계산)
- _ray_intersects_axis_aligned_box : 광선이 축 정렬 상자를 통과하는지 (⑨ 상자 통과 판정)
- _in_zone_3d        : 작업영역 판정 |X|,|Y| ≤ 25, ZONE_Z_LOW ≤ Z ≤ ZONE_Z_HIGH (⑩⑪⑬)
"""

import numpy as np

from .. import settings
from ..geometry import _camera_ray_world, _refract_at_plane


def top_ray_plane_z(pixel, z_world, calib, interface=None):
    if interface is None:
        interface = settings.INTERFACE_TOP
    C, d = _camera_ray_world(pixel, calib)
    refracted = _refract_at_plane(C, d, interface)
    if refracted is None:
        return None
    P, d_ref = refracted
    if abs(d_ref[2]) < 1e-9:
        return None
    lam = (z_world - P[2]) / d_ref[2]
    if lam <= 0:
        return None
    return P + lam * d_ref


def front_ray_plane_y(pixel, y_world, calib, interface=None):
    """Front 굴절 광선과 world Y 평면의 전방 교점."""
    if interface is None:
        interface = settings.INTERFACE_FRONT
    C, d = _camera_ray_world(pixel, calib)
    refracted = _refract_at_plane(C, d, interface)
    if refracted is None:
        return None
    P, d_ref = refracted
    if abs(d_ref[1]) < 1e-9:
        return None
    lam = (float(y_world) - P[1]) / d_ref[1]
    if lam <= 0.0:
        return None
    return P + lam * d_ref


def _ray_intersects_axis_aligned_box(origin, direction, box_min, box_max):
    """전방 광선이 축 정렬 3D box를 통과하는지 slab 방식으로 판정한다."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    box_min = np.asarray(box_min, dtype=float).reshape(3)
    box_max = np.asarray(box_max, dtype=float).reshape(3)

    t_enter = 0.0
    t_exit = np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < box_min[axis] or origin[axis] > box_max[axis]:
                return False
            continue

        t1 = (box_min[axis] - origin[axis]) / direction[axis]
        t2 = (box_max[axis] - origin[axis]) / direction[axis]
        t_near, t_far = min(t1, t2), max(t1, t2)
        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)
        if t_exit < t_enter:
            return False

    return bool(t_exit >= max(t_enter, 0.0))


def _in_zone_3d(x, y, z):
    if not np.all(np.isfinite([x, y, z])):
        return False
    return (
        abs(x) <= settings.ZONE_X_HALF
        and abs(y) <= settings.ZONE_Y_HALF
        and settings.ZONE_Z_LOW <= z <= settings.ZONE_Z_HIGH
    )
