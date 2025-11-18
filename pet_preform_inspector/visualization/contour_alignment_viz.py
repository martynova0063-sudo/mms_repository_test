# visualization/contour_alignment_viz.py
import cv2
import numpy as np
from loguru import logger
from utils import save_debug_image


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    """Вспомогательная функция для рисования пунктирной линии."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    steps = int(length / dash_length)
    for i in range(steps):
        start_x = int(x1 + (x2 - x1) * i / steps)
        start_y = int(y1 + (y2 - y1) * i / steps)
        end_x = int(x1 + (x2 - x1) * (i + 0.5) / steps)
        end_y = int(y1 + (y2 - y1) * (i + 0.5) / steps)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def visualize_contour_alignment(
    original_img,
    selected_contour,
    aligned_approx,
    cx, cy,
    rot_mat,
    angle,
    image_id=None,
    dimensions_data=None,
    pixels_per_mm=None,
    step_flag=False
):
    """
    Визуализация процесса выравнивания контура:
    - исходный контур и габаритные точки,
    - центроид,
    - ось до поворота (пунктир),
    - ось после поворота (сплошная).

    Сохраняется только если step_flag=True.
    """
    if not step_flag:
        return

    img = original_img.copy()
    h, w = img.shape[:2]

    # --- 1. Рисуем исходный контур ---
    cv2.drawContours(img, [selected_contour], -1, (0, 255, 0), 2)

    # --- 2. Габаритные точки на исходном контуре ---
    for pts in [selected_contour]:
        left = tuple(pts[pts[:, :, 0].argmin()][0])
        right = tuple(pts[pts[:, :, 0].argmax()][0])
        top = tuple(pts[pts[:, :, 1].argmin()][0])
        bottom = tuple(pts[pts[:, :, 1].argmax()][0])
        for pt in [left, right, top, bottom]:
            cv2.circle(img, pt, 4, (0, 0, 255), -1)

    # --- 3. Центроид (в исходной системе координат) ---
    if rot_mat is not None:
        inv_rot = cv2.invertAffineTransform(rot_mat)
        centroid_aligned = np.array([[[cx, cy]]])
        centroid_orig = cv2.transform(centroid_aligned, inv_rot)[0, 0]
        cv2.circle(img, (int(centroid_orig[0]), int(centroid_orig[1])), 5, (0, 0, 255), -1)
    else:
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    # --- 4. Ось ДО поворота (пунктирная) ---
    if angle is not None and dimensions_data and pixels_per_mm:
        nominal_len_mm = dimensions_data.get("length", {}).get("nominal", 142.0)
        total_len_px = int((nominal_len_mm + 20.0) * pixels_per_mm)
        half_len = total_len_px // 2
        angle_rad = np.deg2rad(angle)
        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)
        pt1_before = (int(cx - dx), int(cy - dy))
        pt2_before = (int(cx + dx), int(cy + dy))
        draw_dashed_line(img, pt1_before, pt2_before, (0, 255, 255), 2)

    # --- 5. Ось ПОСЛЕ поворота (горизонтальная, сплошная) ---
    if aligned_approx.size > 0:
        x_coords = aligned_approx[:, 0, 0]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        # Центроид в выровненной системе — (cx, cy)
        pt1_after = (int(x_min), int(cy))
        pt2_after = (int(x_max), int(cy))
        # Преобразуем в исходную систему
        if rot_mat is not None:
            inv_rot = cv2.invertAffineTransform(rot_mat)
            start = cv2.transform(np.array([[[x_min, cy]]]), inv_rot)[0, 0]
            end = cv2.transform(np.array([[[x_max, cy]]]), inv_rot)[0, 0]
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 255), 2)

    # --- 6. Сохраняем ---
    save_debug_image(img, "09_aligned_contour_debug", image_id, step_flag=True)