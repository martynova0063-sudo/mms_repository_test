# preprocessing/dense_contour.py
import numpy as np
import cv2


def get_aligned_dense_contour(selected_contour, rot_mat, flip_mat=None):
    """Готовит плотный контур, выровненный и (опционально) перевёрнутый."""
    orig_pts = selected_contour.reshape(-1, 2).astype(np.float32)
    if len(orig_pts) == 0:
        return None

    # Применяем поворот выравнивания
    ones = np.ones((orig_pts.shape[0], 1), dtype=np.float32)
    pts_hom = np.hstack([orig_pts, ones])
    rotated_pts = (rot_mat @ pts_hom.T).T
    aligned_contour = rotated_pts.reshape((-1, 1, 2)).astype(np.int32)

    # Применяем поворот на 180°, если нужно
    if flip_mat is not None:
        ones_flip = np.ones((aligned_contour.shape[0], 1), dtype=np.float32)
        points_hom_flip = np.hstack([aligned_contour.reshape(-1, 2).astype(np.float32), ones_flip])
        flipped_pts = (flip_mat @ points_hom_flip.T).T
        aligned_contour = flipped_pts.reshape((-1, 1, 2)).astype(np.int32)

    return aligned_contour