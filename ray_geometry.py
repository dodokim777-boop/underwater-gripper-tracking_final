"""Ray-plane intersections, ray-box test, and work-zone test used in steps ⑨-⑬."""

import numpy as np

from .. import settings
from ..common.refraction import _camera_ray_world, _refract_at_plane


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
