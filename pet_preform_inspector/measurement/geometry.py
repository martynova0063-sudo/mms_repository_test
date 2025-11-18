# measurement/geometry.py
import cv2
import numpy as np
from sklearn.decomposition import PCA
from loguru import logger


def align_contour(approx, mask):
    """
    Выравнивает контур по главной оси с использованием PCA.

    Args:
        approx: аппроксимированный контур (Nx1x2)
        mask: бинарная маска объекта (HxW)

    Returns:
        aligned_contour: выровненный контур (Nx1x2)
        angle_deg: угол поворота в градусах
        cx, cy: координаты центроида
        rot_mat: матрица аффинного поворота (2x3)
    """
    M = cv2.moments(mask)
    if M["m00"] == 0:
        logger.error("Момент m00 равен 0, невозможно вычислить центроид.")
        return approx, 0.0, 0, 0, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    y_coords, x_coords = np.nonzero(mask)
    points = np.vstack((x_coords, y_coords)).T.astype(np.float32)

    if len(points) < 3:
        logger.error("Недостаточно точек для PCA.")
        return approx, 0.0, cx, cy, None

    pca = PCA(n_components=2).fit(points)
    direction = pca.components_[0]
    dx, dy = direction[0], direction[1]

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    contour_points = approx.reshape(-1, 2).astype(np.float32)
    aligned_points = cv2.transform(np.array([contour_points]), rot_mat)
    aligned_contour = aligned_points.reshape(-1, 1, 2).astype(np.int32)

    return aligned_contour, angle_deg, cx, cy, rot_mat