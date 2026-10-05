"""Step ② (shared): camera ray from a pixel and vector Snell refraction at a planar interface."""

import cv2
import numpy as np

from .. import settings


def _camera_ray_world(pixel, calib):
    R_w, _ = cv2.Rodrigues(calib['rvec']); t_w = calib['tvec'].reshape(3)
    C = -R_w.T @ t_w
    p = np.array([[[float(pixel[0]), float(pixel[1])]]], dtype=np.float32)
    und = cv2.undistortPoints(p, calib['cam_matrix'], calib['dist_coeffs']).reshape(2)
    d_cam = np.array([und[0], und[1], 1.0])
    d = R_w.T @ d_cam
    return C, d / np.linalg.norm(d)


def _refract_at_plane(C, d, interface, n1=None, n2=None):
    if n1 is None:
        n1 = settings.N_AIR
    if n2 is None:
        n2 = settings.N_WATER
    axis = int(interface['axis'])
    value = float(interface['value'])
    normal_to_air = np.asarray(interface['normal_to_air'], dtype=float)
    denom = float(d[axis])
    if abs(denom) < 1e-9:
        return None
    lam = (value - float(C[axis])) / denom
    if lam <= 0.0:
        return None
    P = C + lam * d
    n = normal_to_air / np.linalg.norm(normal_to_air)
    cos_i = -float(np.dot(d, n))
    if cos_i <= 0.0:
        return None
    cos_i = min(cos_i, 1.0)
    eta = float(n1) / float(n2)
    radicand = 1.0 - eta**2 * (1.0 - cos_i**2)
    if radicand < -1e-12:
        return None
    radicand = max(radicand, 0.0)
    d_ref = eta * d + (eta * cos_i - np.sqrt(radicand)) * n
    norm = np.linalg.norm(d_ref)
    if norm < 1e-12:
        return None
    return P, d_ref / norm
