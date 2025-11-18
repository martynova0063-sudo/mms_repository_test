# measurement/measuring_size.py

from typing import Dict, Any
import numpy as np
from utils import scale_to_mm
from loguru import logger
from utils import scale_to_mm
from loguru import logger


def flip_contour_180(contour: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
    if contour.size == 0:
        return contour

    points = contour.reshape(-1, 2).astype(np.float32)
    flipped_points = np.array([
        2 * center_x - points[:, 0],
        2 * center_y - points[:, 1]
    ]).T
    return flipped_points.reshape((-1, 1, 2)).astype(np.int32)


def get_extreme_points(aligned_approx):
    if aligned_approx.size == 0:
        raise ValueError("Контур пустой.")

    x_coords = aligned_approx[:, 0, 0]
    y_coords = aligned_approx[:, 0, 1]

    left_idx = np.argmin(x_coords)
    right_idx = np.argmax(x_coords)
    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)

    return {
        "leftmost": tuple(aligned_approx[left_idx][0]),
        "rightmost": tuple(aligned_approx[right_idx][0]),
        "topmost": tuple(aligned_approx[top_idx][0]),
        "bottommost": tuple(aligned_approx[bottom_idx][0])
    }


def measure_gabarits(aligned_approx, pixels_per_mm=None):
    extreme_points = get_extreme_points(aligned_approx)

    L_px = abs(extreme_points["rightmost"][0] - extreme_points["leftmost"][0])
    Z_px = abs(extreme_points["bottommost"][1] - extreme_points["topmost"][1])

    result = {
        "extreme_points": extreme_points,
        "dimensions_px": {
            "L_px": float(L_px),
            "Z_px": float(Z_px)
        }
    }

    if pixels_per_mm is not None and pixels_per_mm > 0:
        result["dimensions_mm"] = {
            "L_mm": scale_to_mm(L_px, pixels_per_mm),
            "Z_mm": scale_to_mm(Z_px, pixels_per_mm)
        }

    return result


def measure_bottom_diameter(aligned_contour, pixels_per_mm=None, orientation="neck_right"):
    """
    Измеряет диаметр дна (d1), находя первое достижение максимального диаметра от края дна.
    """
    if aligned_contour.size == 0:
        raise ValueError("Контур пустой.")

    points = aligned_contour[:, 0, :]
    x_coords = points[:, 0]

    extreme_points = get_extreme_points(aligned_contour)
    leftmost_x = extreme_points["leftmost"][0]
    rightmost_x = extreme_points["rightmost"][0]
    L_px = abs(rightmost_x - leftmost_x)

    # Настройки
    search_width_pct = 0.15
    min_points = 2
    if pixels_per_mm is None or pixels_per_mm <= 0:
        pixels_per_mm = 1.0

    # Зона поиска
    if orientation == "neck_left":
        start_x = int(round(rightmost_x))
        end_x = int(round(rightmost_x - search_width_pct * L_px))
        step = -1
    else:
        start_x = int(round(leftmost_x))
        end_x = int(round(leftmost_x + search_width_pct * L_px))
        step = 1

    # Коррекция границ
    if (step > 0 and start_x >= end_x) or (step < 0 and start_x <= end_x):
        search_width_pct = 0.25
        if orientation == "neck_left":
            end_x = int(round(rightmost_x - search_width_pct * L_px))
        else:
            end_x = int(round(leftmost_x + search_width_pct * L_px))

    # Генерация X
    if step > 0:
        x_values = list(range(start_x, end_x + 1))
    else:
        x_values = list(range(start_x, end_x - 1, -1))

    # Сбор диаметров
    diameters = {}
    for x in x_values:
        mask_x = np.abs(x_coords - x) <= 0.5
        x_points = points[mask_x]
        if len(x_points) < min_points:
            continue
        top_y = np.max(x_points[:, 1])
        bottom_y = np.min(x_points[:, 1])
        diameter = top_y - bottom_y
        diameters[x] = {"diameter": diameter, "top_y": top_y, "bottom_y": bottom_y}

    if not diameters:
        for x in x_values:
            mask_x = np.abs(x_coords - x) <= 0.5
            x_points = points[mask_x]
            if len(x_points) < 2:
                continue
            top_y = np.max(x_points[:, 1])
            bottom_y = np.min(x_points[:, 1])
            diameter = top_y - bottom_y
            diameters[x] = {"diameter": diameter, "top_y": top_y, "bottom_y": bottom_y}

    if not diameters:
        raise ValueError("Не найдено подходящих сечений для измерения дна.")

    # Поиск первого достижения максимума
    sorted_x = sorted(diameters.keys(), reverse=(step < 0))
    max_diameter = max(diameters[x]["diameter"] for x in diameters)
    best_candidate = None
    for x in sorted_x:
        if abs(diameters[x]["diameter"] - max_diameter) <= 0.1:
            best_candidate = x
            break

    if best_candidate is None:
        best_candidate = max(diameters.keys(), key=lambda x: diameters[x]["diameter"])

    # Результат
    result_diam = diameters[best_candidate]
    d1_px = result_diam["diameter"]
    top_pt = (int(best_candidate), int(result_diam["top_y"]))
    bottom_pt = (int(best_candidate), int(result_diam["bottom_y"]))

    result = {
        "bottom_points": {"top_pt": top_pt, "bottom_pt": bottom_pt},
        "dimensions_px": {"d1_px": float(d1_px)},
        "x_level": float(best_candidate)
    }

    if pixels_per_mm is not None and pixels_per_mm > 0:
        d1_mm_val = scale_to_mm(d1_px, pixels_per_mm)
        result["dimensions_mm"] = {"d1_mm": d1_mm_val}
    else:
        d1_mm_val = d1_px

    logger.debug(f"d1 = {d1_mm_val:.2f} мм (ориентация: {orientation})")
    return result


def measure_straightness_deviation(
    aligned_contour: np.ndarray,
    pixels_per_mm: float = 1.0,
    neck_percentage: float = 0.15
) -> Dict[str, Any]:
    """
    Измеряет:
    - S_max — максимальное отклонение оси от прямой,
    - S_rms — среднеквадратичное отклонение (RMS).
    Исключает горлышко (последние neck_percentage длины).
    """
    if aligned_contour.size == 0:
        raise ValueError("Контур пустой.")

    points = aligned_contour.reshape(-1, 2).astype(np.float32)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    L_px = max_x - min_x

    # Исключаем горлышко
    neck_start_x = max_x - neck_percentage * L_px

    num_slices = max(5, int((neck_start_x - min_x) / 15))   # 15 ширина сегмента
    step = (neck_start_x - min_x) / num_slices

    slice_centers = []
    for i in range(num_slices + 1):
        x_slice = min_x + i * step
        mask = np.abs(x_coords - x_slice) <= 2.0
        y_slice = y_coords[mask]
        if len(y_slice) > 0:
            top_y = np.max(y_slice)
            bottom_y = np.min(y_slice)
            y_center = (top_y + bottom_y) / 2.0
            slice_centers.append((x_slice, y_center))

    if len(slice_centers) < 3:
        raise ValueError("Недостаточно сечений для аппроксимации прямой.")

    slice_centers = np.array(slice_centers)

    # Аппроксимация прямой
    A = np.vstack([slice_centers[:, 0], np.ones(len(slice_centers))]).T
    a, b = np.linalg.lstsq(A, slice_centers[:, 1], rcond=None)[0]

    # Расстояния до прямой
    distances = np.abs(a * slice_centers[:, 0] - slice_centers[:, 1] + b) / np.sqrt(a*a + 1)

    # Максимальное отклонение
    max_idx = np.argmax(distances)
    max_deviation_px = float(distances[max_idx])
    max_point = tuple(slice_centers[max_idx].astype(int))

    # Среднеквадратичное отклонение (RMS)
    rms_deviation_px = float(np.sqrt(np.mean(distances ** 2)))

    result = {
        "straightness_deviation_px": max_deviation_px,
        "straightness_deviation_mm": scale_to_mm(max_deviation_px, pixels_per_mm),
        "straightness_rms_mm": scale_to_mm(rms_deviation_px, pixels_per_mm),
        "max_deviation_point": max_point,
        "line_coefficients": {"a": float(a), "b": float(b)},
        "slice_centers": slice_centers.tolist(),
        "neck_start_x": float(neck_start_x),
    }

    logger.debug(
        f"Прямолинейность: S_max={result['straightness_deviation_mm']:.3f} мм, "
        f"S_rms={result['straightness_rms_mm']:.3f} мм (исключено {neck_percentage*100}% горлышка)"
    )
    return result