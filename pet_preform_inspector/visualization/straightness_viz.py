# visualization/straightness_viz.py

import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path
import config
from loguru import logger


def visualize_straightness(
    original_img: np.ndarray,
    selected_contour: np.ndarray,
    straightness_data: Dict[str, Any],
    rot_mat: np.ndarray,
    image_id: str,
    pixels_per_mm: float,
    dimensions_data: Dict[str, Any],
    background_color=(128, 128, 128)
):
    """
    Визуализация прямолинейности преформы на основе аппроксимации центральной оси.
    Исключает горлышко из измерения (последние 20% длины).

    Ожидает, что straightness_data содержит:
    - 'line_coefficients': {'a': float, 'b': float}
    - 'slice_centers': List[Tuple[float, float]]
    - 'max_deviation_point': Tuple[int, int]
    - 'straightness_deviation_mm': float
    - 'neck_start_x': float (опционально)
    """
    h, w = original_img.shape[:2]
    vis = np.full((h, w, 3), background_color, dtype=np.uint8)

    # Повернуть изображение и контур
    rotated_img = cv2.warpAffine(original_img, rot_mat, (w, h), borderValue=background_color)
    contour_pts = selected_contour.reshape(-1, 2).astype(np.float32)
    ones = np.ones((contour_pts.shape[0], 1), dtype=np.float32)
    hom_pts = np.hstack([contour_pts, ones])
    rotated_pts = (rot_mat @ hom_pts.T).T
    rotated_contour = rotated_pts.reshape((-1, 1, 2)).astype(np.int32)

    # Наложить объект
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [rotated_contour], -1, 255, thickness=cv2.FILLED)
    obj = cv2.bitwise_and(rotated_img, rotated_img, mask=mask)
    vis[mask > 0] = obj[mask > 0]

    # Рисуем аппроксимирующую прямую и точки центров
    line_coef = straightness_data.get("line_coefficients")
    slice_centers = straightness_data.get("slice_centers", [])

    if line_coef is not None and slice_centers:
        a = line_coef["a"]
        b = line_coef["b"]
        centers = np.array(slice_centers)

        # Границы для прямой (только по телу)
        x_min = int(np.min(centers[:, 0]))
        x_max = int(np.max(centers[:, 0]))
        y1 = int(a * x_min + b)
        y2 = int(a * x_max + b)

        y1 = np.clip(y1, 0, h - 1)
        y2 = np.clip(y2, 0, h - 1)

        # Жёлтая осевая линия
        cv2.line(vis, (x_min, y1), (x_max, y2), (0, 255, 255), 2)

        # Зелёные точки центров сечений
        for x, y in centers:
            cv2.circle(vis, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    # === ГРАНИЦА ГОРЛЫШКА ===
    if "neck_start_x" in straightness_data:
        neck_x = int(straightness_data["neck_start_x"])
        # Вертикальная линия
        cv2.line(vis, (neck_x, 0), (neck_x, h), (255, 255, 0), 1, lineType=cv2.LINE_AA)
        # Подпись
        cv2.putText(
            vis, "Neck start", (neck_x + 5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
        )

    # Точка максимального отклонения
    max_pt = straightness_data.get("max_deviation_point")
    if max_pt is not None:
        cv2.circle(vis, tuple(max_pt), radius=6, color=(0, 0, 255), thickness=-1)

        # Стрелка от прямой к точке
        if line_coef is not None:
            x_mp, y_mp = max_pt
            a, b = line_coef["a"], line_coef["b"]
            y_on_line = a * x_mp + b
            proj_pt = (int(x_mp), int(y_on_line))
            cv2.arrowedLine(
                vis, proj_pt, (int(x_mp), int(y_mp)),
                (255, 0, 255), 2, tipLength=0.05
            )


    # Текст с результатом
    s_max_mm = straightness_data.get("straightness_deviation_mm", 0.0)
    s_rms_mm = straightness_data.get("straightness_rms_mm", 0.0)

    text1 = f"S_max = {s_max_mm:.3f} mm"
    text2 = f"S_rms  = {s_rms_mm:.3f} mm"

    cv2.putText(vis, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Сохранение
    config.DEBUG_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    name = Path(image_id).stem
    output_path = config.DEBUG_OUTPUT_FOLDER / f"{name}_10_straightness.jpg"
    cv2.imwrite(str(output_path), vis)
    logger.debug(f"Сохранена визуализация прямолинейности: {output_path}")