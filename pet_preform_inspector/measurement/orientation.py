# measurement/orientation.py

import numpy as np
import cv2
from loguru import logger


def apply_180_flip(contour: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Поворачивает контур на 180° вокруг точки (cx, cy)."""
    if contour.size == 0:
        logger.warning("Попытка перевернуть пустой контур в apply_180_flip")
        return contour

    points = contour.reshape(-1, 2).astype(np.float32)
    flipped_points = np.array([
        2 * cx - points[:, 0],
        2 * cy - points[:, 1]
    ]).T

    logger.debug(f"Применён поворот на 180° вокруг ({cx:.2f}, {cy:.2f})")
    return flipped_points.reshape((-1, 1, 2)).astype(np.int32)


def ensure_correct_orientation(aligned_approx, cx_fallback: float, cy_fallback: float, image_id: str = "unknown"):
    """
    (Оставлено для совместимости, но больше не используется в основном анализе)
    """
    from measurement.measuring_size import get_extreme_points

    if aligned_approx.size == 0:
        logger.error(f"[{image_id}] Пустой контур передан в ensure_correct_orientation")
        raise ValueError("Пустой контур в ensure_correct_orientation")

    M = cv2.moments(aligned_approx)
    if M["m00"] == 0:
        logger.warning(f"[{image_id}] Не удалось вычислить моменты контура. Используем fallback центр масс.")
        cx_actual = cx_fallback
        cy_actual = cy_fallback
    else:
        cx_actual = M["m10"] / M["m00"]
        cy_actual = M["m01"] / M["m00"]
        logger.debug(f"[{image_id}] Центр масс выровненного контура: ({cx_actual:.2f}, {cy_actual:.2f})")

    try:
        extreme_points = get_extreme_points(aligned_approx)
        topmost_x = extreme_points["topmost"][0]
        bottommost_x = extreme_points["bottommost"][0]
        logger.debug(f"[{image_id}] Topmost X: {topmost_x:.2f}")
        logger.debug(f"[{image_id}] Bottommost X: {bottommost_x:.2f}")
    except Exception as e:
        logger.error(f"[{image_id}] Ошибка при получении габаритных точек: {e}")
        raise

    if topmost_x < cx_actual and bottommost_x < cx_actual:
        logger.info(f"[{image_id}] 🔁 Горлышко слева — применяем поворот на 180°")

        flipped_contour = apply_180_flip(aligned_approx, cx_actual, cy_actual)

        M_flipped = cv2.moments(flipped_contour)
        if M_flipped["m00"] == 0:
            logger.warning(f"[{image_id}] Моменты после переворота нулевые. Используем предыдущий центр.")
            new_cx = cx_actual
            new_cy = cy_actual
        else:
            new_cx = M_flipped["m10"] / M_flipped["m00"]
            new_cy = M_flipped["m01"] / M_flipped["m00"]
            logger.debug(f"[{image_id}] Новый центр масс после переворота: ({new_cx:.2f}, {new_cy:.2f})")

        return flipped_contour, True, new_cx, new_cy

    elif topmost_x > cx_actual and bottommost_x > cx_actual:
        logger.info(f"[{image_id}] ✅ Горлышко справа — ориентация корректна")
        return aligned_approx, False, cx_actual, cy_actual

    else:
        logger.warning(f"[{image_id}] ⚖️ Смешанное положение — ориентация не изменена")
        return aligned_approx, False, cx_actual, cy_actual


def determine_neck_orientation(
    extreme_points: dict,
    cx: float,
    cy: float,
    image_id: str = "unknown"
) -> str:
    """
    Определяет, с какой стороны находится горлышко преформы.

    Возвращает:
        "neck_right" — горлышко справа (дно слева)
        "neck_left"  — горлышко слева (дно справа)
    """
    topmost_x = extreme_points["topmost"][0]
    bottommost_x = extreme_points["bottommost"][0]

    logger.debug(f"[{image_id}] Определение ориентации: topmost_x={topmost_x:.2f}, bottommost_x={bottommost_x:.2f}, cx={cx:.2f}")

    if topmost_x < cx and bottommost_x < cx:
        orientation = "neck_left"
        logger.info(f"[{image_id}] 🧭 Ориентация определена: горлышко СЛЕВА")
    else:
        orientation = "neck_right"
        logger.info(f"[{image_id}] 🧭 Ориентация определена: горлышко СПРАВА")

    return orientation