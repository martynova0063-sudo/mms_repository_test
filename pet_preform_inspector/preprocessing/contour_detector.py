# preprocessing/contour_detector.py

import cv2
import numpy as np
from loguru import logger
from config import (
    MIN_CONTOUR_AREA,
    EPSILON_FACTOR,
    BINARY_METHOD,
    STEP_1_DETECT_CONTOUR,
    GAUSSIAN_BLUR_KERNEL_SIZE,
    GAUSSIAN_BLUR_SIGMA_X,
    CANNY_THRESHOLD1,
    CANNY_THRESHOLD2,
    MORPHOLOGY_KERNEL_SIZE,
    OTSU_MORPHOLOGY_KERNEL_SIZE
)
from utils import save_debug_image

def _binarize_otsu():
    1==1

def _binarize_background_subtraction_canny():
    1==1

def _binarize_distance_transform():
    1==1    

def is_elongated_contour(contour, min_aspect_ratio=2.0):
    """
    Фильтр: оставляет только вытянутые объекты (независимо от ориентации).
    Использует ориентированный bounding box (minAreaRect).
    """
    if len(contour) < 5:
        return False

    rect = cv2.minAreaRect(contour)
    _, (width, height), angle = rect

    if width <= 0 or height <= 0:
        return False

    aspect_ratio = max(width, height) / min(width, height)
    return aspect_ratio >= min_aspect_ratio


def is_contour_touching_border(contour, img_shape, border_margin=2):
    """
    Проверяет, касается ли контур границ изображения (с запасом в border_margin пикселей).
    """
    h, w = img_shape[:2]
    x_coords = contour[:, 0, 0]
    y_coords = contour[:, 0, 1]

    if (
        np.any(x_coords <= border_margin) or
        np.any(x_coords >= w - 1 - border_margin) or
        np.any(y_coords <= border_margin) or
        np.any(y_coords >= h - 1 - border_margin)
    ):
        return True
    return False


def apply_morphology(binary, kernel_size):
    """Морфологическая очистка: закрытие + открытие + дилатация"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def detect_contour(img, image_id=None):
    """
    Обнаруживает контур преформы и возвращает его вместе с бинарной маской.
    Если преформа не подходит (обрезана, мала, не вытянута) — возвращает (None, None, None).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if STEP_1_DETECT_CONTOUR:
        save_debug_image(gray, "01_gray", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X)
    if STEP_1_DETECT_CONTOUR:
        save_debug_image(blurred, "02_blurred", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    # === ВЫБОР МЕТОДА БИНАРИЗАЦИИ ===
    if BINARY_METHOD == "CANNY":
        # Чистый Canny — без заполнения, но с замыканием
        edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

        # Замыкаем разрывы морфологией
        kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
        binary_intermediate = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if STEP_1_DETECT_CONTOUR:
            save_debug_image(binary_intermediate, "06_binary_canny", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    elif BINARY_METHOD == "OTSU":
        thresh = _binarize_otsu(blurred, OTSU_MORPHOLOGY_KERNEL_SIZE)
        binary_intermediate = apply_morphology(thresh, MORPHOLOGY_KERNEL_SIZE)
        if STEP_1_DETECT_CONTOUR:
            save_debug_image(binary_intermediate, "06_binary_otsu", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    elif BINARY_METHOD == "BACKGROUND_SUB_CANNY":
        binary_intermediate = _binarize_background_subtraction_canny(img)
        binary_intermediate = apply_morphology(binary_intermediate, MORPHOLOGY_KERNEL_SIZE)
        binary_intermediate = cv2.GaussianBlur(binary_intermediate, (5, 5), 0)
        if STEP_1_DETECT_CONTOUR:
            save_debug_image(binary_intermediate, "06_binary_background_sub_canny", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    elif BINARY_METHOD == "DISTANCE_TRANSFORM":
        binary_intermediate = _binarize_distance_transform(img)
        if STEP_1_DETECT_CONTOUR:
            save_debug_image(binary_intermediate, "06_binary_distance_transform", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    else:
        logger.warning(f"[{image_id}] Неизвестный метод бинаризации: {BINARY_METHOD}")
        return None, None, None

    # Поиск контуров
    contours, _ = cv2.findContours(binary_intermediate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    logger.debug(f"[{image_id}] Найдено контуров: {len(contours)}")

    # Фильтрация
    valid_contours = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        reason = []

        if area <= MIN_CONTOUR_AREA:
            reason.append(f"площадь={area:.0f} <= {MIN_CONTOUR_AREA}")

        if not is_elongated_contour(c):
            if len(c) >= 5:
                rect = cv2.minAreaRect(c)
                _, (w, h), _ = rect
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                reason.append(f"aspect={aspect:.2f} < 2.0")
            else:
                reason.append("слишком мало точек для minAreaRect")

        if is_contour_touching_border(c, img.shape, border_margin=2):
            h_img, w_img = img.shape[:2]
            x_coords = c[:, 0, 0]
            y_coords = c[:, 0, 1]
            touches = []
            if np.any(x_coords <= 2):
                touches.append("left")
            if np.any(x_coords >= w_img - 1 - 2):
                touches.append("right")
            if np.any(y_coords <= 2):
                touches.append("top")
            if np.any(y_coords >= h_img - 1 - 2):
                touches.append("bottom")
            reason.append(f"касается края: {', '.join(touches)}")

        if not reason:
            valid_contours.append(c)
        else:
            logger.debug(f"[{image_id}] Контур #{i} отбракован: {'; '.join(reason)}")

    logger.debug(f"[{image_id}] Валидных контуров после фильтрации: {len(valid_contours)}")

    if not valid_contours:
        logger.warning(f"[{image_id}] Отбраковка: ни один контур не прошёл фильтрацию")
        return None, None, None

    selected_contour = max(valid_contours, key=cv2.contourArea)
    logger.info(f"[{image_id}] Выбран контур с площадью {cv2.contourArea(selected_contour):.0f}")

    # Отладочная визуализация
    if STEP_1_DETECT_CONTOUR:
        all_contours_img = img.copy()
        for c in contours:
            area = cv2.contourArea(c)
            passes_area = area > MIN_CONTOUR_AREA
            passes_shape = is_elongated_contour(c)
            passes_border = not is_contour_touching_border(c, img.shape, border_margin=2)
            color = (0, 255, 0) if (passes_area and passes_shape and passes_border) else (0, 0, 255)
            cv2.drawContours(all_contours_img, [c], -1, color, 2)

        leftmost = tuple(selected_contour[selected_contour[:, :, 0].argmin()][0])
        rightmost = tuple(selected_contour[selected_contour[:, :, 0].argmax()][0])
        topmost = tuple(selected_contour[selected_contour[:, :, 1].argmin()][0])
        bottommost = tuple(selected_contour[selected_contour[:, :, 1].argmax()][0])

        cv2.circle(all_contours_img, leftmost, 3, (0, 0, 255), -1)
        cv2.circle(all_contours_img, rightmost, 3, (0, 0, 255), -1)
        cv2.circle(all_contours_img, topmost, 3, (0, 0, 255), -1)
        cv2.circle(all_contours_img, bottommost, 3, (0, 0, 255), -1)

        save_debug_image(all_contours_img, "07_all_contours", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    # Аппроксимация и маска
    epsilon = EPSILON_FACTOR * cv2.arcLength(selected_contour, True)
    approx = cv2.approxPolyDP(selected_contour, epsilon, True)

    # Маска — тонкая линия по контуру (не заполненная область!)
    mask_filled = np.zeros_like(gray)
    cv2.drawContours(mask_filled, [selected_contour], -1, 255, thickness=1)

    if STEP_1_DETECT_CONTOUR:
        save_debug_image(mask_filled, "08_mask_filled", image_id, step_flag=STEP_1_DETECT_CONTOUR)

    return approx, mask_filled, selected_contour