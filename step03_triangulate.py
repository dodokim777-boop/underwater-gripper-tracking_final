"""Step ③: closest-point triangulation of the top and front rays (refraction ON and OFF)."""

from dataclasses import dataclass

import numpy as np

from .. import settings
from ..common.refraction import _camera_ray_world, _refract_at_plane


@dataclass(frozen=True)
class TriangulationResult:
    point_world: np.ndarray
    ray_gap_cm: float
    s_top_cm: float
    s_front_cm: float


def _closest_points_on_forward_rays(P1, u1, P2, u2):
    w0 = P1 - P2
    b = float(np.dot(u1, u2))
    d1 = float(np.dot(u1, w0))
    e1 = float(np.dot(u2, w0))
    denom = 1.0 - b * b
    if abs(denom) < 1e-9:
        return None
    s = (b * e1 - d1) / denom
    t = (e1 - b * d1) / denom
    if s <= 0.0 or t <= 0.0:
        return None
    Q1 = P1 + s * u1
    Q2 = P2 + t * u2
    return TriangulationResult(
        0.5 * (Q1 + Q2),
        float(np.linalg.norm(Q1 - Q2)),
        float(s),
        float(t),
    )


def triangulate_pair(
    pt_top, pt_front, calib_top, calib_front, apply_refraction
):
    C_t, d_t = _camera_ray_world(pt_top, calib_top)
    C_f, d_f = _camera_ray_world(pt_front, calib_front)

    if apply_refraction:
        ray_top = _refract_at_plane(C_t, d_t, settings.INTERFACE_TOP)
        ray_front = _refract_at_plane(C_f, d_f, settings.INTERFACE_FRONT)
        if ray_top is None or ray_front is None:
            return None
    else:
        ray_top = (C_t, d_t)
        ray_front = (C_f, d_f)

    return _closest_points_on_forward_rays(
        ray_top[0], ray_top[1], ray_front[0], ray_front[1]
    )


def triangulate_on_off(pt_top, pt_front, calib_top, calib_front):
    tri_on = triangulate_pair(
        pt_top, pt_front, calib_top, calib_front, apply_refraction=True
    )
    tri_off = triangulate_pair(
        pt_top, pt_front, calib_top, calib_front, apply_refraction=False
    )
    geometry_status_on = 'VALID' if tri_on is not None else 'FAILED'
    geometry_status_off = 'VALID' if tri_off is not None else 'FAILED'
    return tri_on, tri_off, geometry_status_on, geometry_status_off
