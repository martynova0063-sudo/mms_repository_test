# visualization/annotated_result_viz.py

import cv2
import numpy as np
from loguru import logger
from utils import save_debug_image, scale_to_mm


# =============================
# === КОНФИГУРАЦИЯ СТИЛЕЙ И ПАРАМЕТРОВ ===
# =============================
DEFAULT_CONFIG = {
    # --- Цвета ---
    "colors": {
        "contour": (0, 255, 0),
        "center_point": (0, 0, 255),
        "gabarit_points": (0, 0, 255),
        "bottom_points": (255, 0, 255),
        "dimension_line": (255, 255, 255),
        "extension_line": (255, 255, 255),
        "text": (255, 255, 255),
        "background": (128, 128, 128),
    },
    # --- Толщины линий ---
    "thicknesses": {
        "contour": 2,
        "dimension_line": 2,
        "extension_line": 1,
        "arrow": 2,
    },
    # --- Радиусы точек ---
    "radii": {
        "center_point": 5,
        "gabarit_points": 4,
        "bottom_points": 3,
    },
    # --- Параметры текста ---
    "text": {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "font_scale": 0.5,
        "thickness": 1,
    },
    # --- Параметры осевой линии ---
    "axis_line": {
        "dash_length": 10,
    },
    # --- Параметры стрелок ---
    "arrow": {
        "length_px": 5,
    },
    # --- Настройки размеров ---
    "dimensions": {
        "L": {
            "direction": "horizontal",
            "offset_mm": 25.0,
            "overhang_mm": 2.0,
            "text_offset_mm": -2.0,
            "text_vertical": False,
            "arrow_direction": "inward",
            "extension_direction": "down",
            "text_offset_x_mm": 0.0,
        },
        "Z": {
            "direction": "vertical",
            "offset_mm": 20.0,
            "overhang_mm": 2.0,
            "text_offset_mm": 30.0,
            "text_vertical": True,
            "arrow_direction": "inward",
            "extension_direction": "auto",
            "text_offset_x_mm": -2.0,
        },
        "d1": {
            "direction": "vertical",
            "offset_mm": 25.0,
            "overhang_mm": 1.5,
            "text_offset_mm": -2.0,
            "text_vertical": True,
            "arrow_direction": "outward",
            "extension_direction": "auto",
            "text_offset_x_mm": 0.0,
            "side": "left",  # может быть "left" или "right"
        }
    }
}


def draw_rotated_text(
    img,
    text,
    position,
    is_vertical=False,
    color=(255, 255, 255),
    background=(128, 128, 128),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    thickness=1,
):
    """Рисует текст с фоном и поворотом."""
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size

    canvas = np.full((text_h, text_w, 3), background, dtype=np.uint8)
    cv2.putText(canvas, text, (0, text_h - 1), font, font_scale, color, thickness)

    h_img, w_img = img.shape[:2]

    if is_vertical:
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_rot, w_rot = canvas.shape[:2]
        x, y = position
        x = max(0, min(x, w_img - w_rot))
        y = max(0, min(y - h_rot, h_img - h_rot))
        img[y:y + h_rot, x:x + w_rot] = canvas
    else:
        x, y = position
        x = max(0, min(x - text_w // 2, w_img - text_w))
        y = max(0, min(y - text_h // 2, h_img - text_h))
        cv2.rectangle(img, (x, y), (x + text_w, y + text_h), background, -1)
        cv2.putText(img, text, (x, y + text_h), font, font_scale, color, thickness)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Рисует пунктирную линию."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    steps = max(1, int(length / dash_length))
    for i in range(steps):
        start_x = int(x1 + (x2 - x1) * i / steps)
        start_y = int(y1 + (y2 - y1) * i / steps)
        end_x = int(x1 + (x2 - x1) * (i + 0.5) / steps)
        end_y = int(y1 + (y2 - y1) * (i + 0.5) / steps)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def draw_dimension_universal(
    img,
    pt1,
    pt2,
    label,
    config,
    pixels_per_mm,
    dimension_key,
    neck_orientation="neck_right",
):
    """
    Универсальная отрисовка размера по конфигурации.
    neck_orientation: "neck_right" → дно слева; "neck_left" → дно справа
    """
    dim_cfg = config["dimensions"][dimension_key]
    colors = config["colors"]
    thicknesses = config["thicknesses"]
    text_cfg = config["text"]
    arrow_len = config["arrow"]["length_px"]

    direction = dim_cfg["direction"]
    offset_px = int(dim_cfg["offset_mm"] * pixels_per_mm)
    overhang_px = int(dim_cfg["overhang_mm"] * pixels_per_mm)
    text_offset_px = int(dim_cfg["text_offset_mm"] * pixels_per_mm)
    arrow_direction = dim_cfg.get("arrow_direction", "outward")
    extension_direction = dim_cfg.get("extension_direction", "auto")

    if direction == "vertical":
        if dimension_key == "d1":
            d1_side = dim_cfg.get("side", "left")
            if d1_side == "left":
                line_x = min(pt1[0], pt2[0]) - offset_px
            else:  # "right"
                line_x = max(pt1[0], pt2[0]) + offset_px
        elif dimension_key == "Z":
            # Для Z: если горлышко слева — рисуем слева, иначе — справа
            if neck_orientation == "neck_left":
                line_x = min(pt1[0], pt2[0]) - offset_px  # слева
            else:
                line_x = max(pt1[0], pt2[0]) + offset_px  # справа
        else:
            line_x = max(pt1[0], pt2[0]) + offset_px  # по умолчанию справа
        line_start = (line_x, pt1[1])
        line_end = (line_x, pt2[1])

        # Выносные линии
        if dimension_key == "d1":
            d1_side = dim_cfg.get("side", "left")
            if extension_direction == "auto":
                ext_x1 = line_x - overhang_px if d1_side == "left" else line_x + overhang_px
                ext_x2 = line_x - overhang_px if d1_side == "left" else line_x + overhang_px
            elif extension_direction == "left":
                ext_x1 = line_x - overhang_px
                ext_x2 = line_x - overhang_px
            elif extension_direction == "right":
                ext_x1 = line_x + overhang_px
                ext_x2 = line_x + overhang_px
            else:
                ext_x1 = line_x
                ext_x2 = line_x
        elif dimension_key == "Z":
            if extension_direction == "auto":
                ext_x1 = line_x - overhang_px if neck_orientation == "neck_left" else line_x + overhang_px
                ext_x2 = line_x - overhang_px if neck_orientation == "neck_left" else line_x + overhang_px
            elif extension_direction == "left":
                ext_x1 = line_x - overhang_px
                ext_x2 = line_x - overhang_px
            elif extension_direction == "right":
                ext_x1 = line_x + overhang_px
                ext_x2 = line_x + overhang_px
            else:
                ext_x1 = line_x
                ext_x2 = line_x
        else:
            if extension_direction == "auto":
                ext_x1 = line_x + overhang_px
                ext_x2 = line_x + overhang_px
            elif extension_direction == "left":
                ext_x1 = line_x - overhang_px
                ext_x2 = line_x - overhang_px
            elif extension_direction == "right":
                ext_x1 = line_x + overhang_px
                ext_x2 = line_x + overhang_px
            else:
                ext_x1 = line_x
                ext_x2 = line_x

        cv2.line(img, pt1, (ext_x1, pt1[1]), colors["extension_line"], thicknesses["extension_line"])
        cv2.line(img, pt2, (ext_x2, pt2[1]), colors["extension_line"], thicknesses["extension_line"])

        # Размерная линия
        cv2.line(img, line_start, line_end, colors["dimension_line"], thicknesses["dimension_line"])

        # Стрелки
        if arrow_direction == "inward":
            cv2.line(img, line_start, (line_start[0] - arrow_len, line_start[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_start, (line_start[0] + arrow_len, line_start[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] - arrow_len, line_end[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] + arrow_len, line_end[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
        else:  # outward
            cv2.line(img, line_start, (line_start[0] - arrow_len, line_start[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_start, (line_start[0] + arrow_len, line_start[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] - arrow_len, line_end[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] + arrow_len, line_end[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])

        # Текст
        text_offset_x_px = int(dim_cfg.get("text_offset_x_mm", 0.0) * pixels_per_mm)
        if dimension_key == "d1":
            d1_side = dim_cfg.get("side", "left")
            text_x = line_start[0] - 20 + text_offset_x_px if d1_side == "left" else line_start[0] + 20 + text_offset_x_px
        elif dimension_key == "Z":
            text_x = line_start[0] - 10 + text_offset_x_px if neck_orientation == "neck_left" else line_start[0] + 10 + text_offset_x_px
        else:
            text_x = line_start[0] - 10 + text_offset_x_px
        text_y = line_start[1] + text_offset_px
        draw_rotated_text(
            img, label, (text_x, text_y),
            is_vertical=dim_cfg["text_vertical"],
            color=colors["text"],
            background=colors["background"],
            font=text_cfg["font"],
            font_scale=text_cfg["font_scale"],
            thickness=text_cfg["thickness"]
        )

    elif direction == "horizontal":
        line_y = max(pt1[1], pt2[1]) + offset_px
        line_start = (pt1[0], line_y)
        line_end = (pt2[0], line_y)

        # Выносные линии
        if extension_direction == "auto":
            ext_y1 = line_start[1] - overhang_px
            ext_y2 = line_end[1] - overhang_px
        elif extension_direction == "down":
            ext_y1 = line_start[1] + overhang_px
            ext_y2 = line_end[1] + overhang_px
        elif extension_direction == "up":
            ext_y1 = line_start[1] - overhang_px
            ext_y2 = line_end[1] - overhang_px
        else:
            ext_y1 = line_start[1]
            ext_y2 = line_end[1]

        cv2.line(img, pt1, (pt1[0], ext_y1), colors["extension_line"], thicknesses["extension_line"])
        cv2.line(img, pt2, (pt2[0], ext_y2), colors["extension_line"], thicknesses["extension_line"])

        # Размерная линия
        cv2.line(img, line_start, line_end, colors["dimension_line"], thicknesses["dimension_line"])

        # Стрелки
        if arrow_direction == "inward":
            cv2.line(img, line_start, (line_start[0] + arrow_len, line_start[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_start, (line_start[0] + arrow_len, line_start[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] - arrow_len, line_end[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] - arrow_len, line_end[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
        else:  # outward
            cv2.line(img, line_start, (line_start[0] - arrow_len, line_start[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_start, (line_start[0] - arrow_len, line_start[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] + arrow_len, line_end[1] - arrow_len), colors["dimension_line"], thicknesses["arrow"])
            cv2.line(img, line_end, (line_end[0] + arrow_len, line_end[1] + arrow_len), colors["dimension_line"], thicknesses["arrow"])

        # Текст
        text_offset_x_px = int(dim_cfg.get("text_offset_x_mm", 0.0) * pixels_per_mm)
        text_x = (line_start[0] + line_end[0]) // 2 + text_offset_x_px
        text_y = line_start[1] + text_offset_px
        draw_rotated_text(
            img, label, (text_x, text_y),
            is_vertical=dim_cfg["text_vertical"],
            color=colors["text"],
            background=colors["background"],
            font=text_cfg["font"],
            font_scale=text_cfg["font_scale"],
            thickness=text_cfg["thickness"]
        )


def visualize_dimensioned_drawing(
    original_img,
    selected_contour,
    cx, cy,
    angle,
    rot_mat,
    image_id,
    dimensions_data,
    pixels_per_mm,
    gabarit_data,
    neck_data=None,
    bottom_data=None,
    neck_orientation="neck_right",  # "neck_right" → дно слева; "neck_left" → дно справа
    config=None,
    step_flag=False
):
    """
    Визуализирует преформу с размерами L, Z, d1.
    Сохраняется только если step_flag=True.
    """
    if not step_flag:
        return

    config = config or DEFAULT_CONFIG
    colors = config["colors"]
    thicknesses = config["thicknesses"]
    radii = config["radii"]
    text_cfg = config["text"]

    h, w = original_img.shape[:2]
    background_color = colors["background"]
    result_img = np.full((h, w, 3), background_color, dtype=np.uint8)

    # --- Повернуть изображение и контур ---
    rotated_img = cv2.warpAffine(original_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=background_color)
    contour_points = selected_contour.reshape(-1, 2).astype(np.float32)
    ones = np.ones((contour_points.shape[0], 1), dtype=np.float32)
    points_hom = np.hstack([contour_points, ones])
    rotated_points = (rot_mat @ points_hom.T).T
    rotated_contour = rotated_points.reshape((-1, 1, 2)).astype(np.int32)

    if len(rotated_contour) == 0:
        logger.error(f"[{image_id}] Пустой контур после поворота.")
        return result_img

    # --- Центр масс повёрнутого контура ---
    M = cv2.moments(rotated_contour)
    if M["m00"] != 0:
        cx_rot = int(M["m10"] / M["m00"])
        cy_rot = int(M["m01"] / M["m00"])
    else:
        x_coords = rotated_contour[:, 0, 0]
        y_coords = rotated_contour[:, 0, 1]
        cx_rot = int((np.min(x_coords) + np.max(x_coords)) / 2)
        cy_rot = int((np.min(y_coords) + np.max(y_coords)) / 2)

    # --- Маска и объект ---
    mask_rotated = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_rotated, [rotated_contour], -1, 255, thickness=cv2.FILLED)
    object_region = cv2.bitwise_and(rotated_img, rotated_img, mask=mask_rotated)

    # --- Сдвиг в центр изображения ---
    center_x, center_y = w // 2, h // 2
    dx = center_x - cx_rot
    dy = center_y - cy_rot

    translation_mat = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_object = cv2.warpAffine(object_region, translation_mat, (w, h), borderValue=background_color)
    shifted_mask = cv2.warpAffine(mask_rotated, translation_mat, (w, h), borderValue=0)

    # --- Вставляем объект на фон ---
    result_img[shifted_mask > 0] = shifted_object[shifted_mask > 0]

    # --- Сдвинутый контур ---
    shifted_contour = rotated_contour.astype(np.float32)
    shifted_contour[:, :, 0] += dx
    shifted_contour[:, :, 1] += dy
    shifted_contour = shifted_contour.astype(np.int32)

    # --- Рисуем контур ---
    cv2.drawContours(result_img, [shifted_contour], -1, colors["contour"], thicknesses["contour"])

    # --- Центр масс ---
    cv2.circle(result_img, (center_x, center_y), radii["center_point"], colors["center_point"], -1)

    # --- Осевая линия ---
    nominal_length_mm = dimensions_data.get("length", {}).get("nominal", 142.0)
    total_line_length_px = int((nominal_length_mm + 20.0) * pixels_per_mm)
    half_line = total_line_length_px // 2
    pt1 = (center_x - half_line, center_y)
    pt2 = (center_x + half_line, center_y)
    draw_dashed_line(
        result_img, pt1, pt2,
        color=colors["dimension_line"],
        thickness=1,
        dash_length=config["axis_line"]["dash_length"]
    )

    # --- Габаритные точки ---
    extreme_points = gabarit_data["extreme_points"]
    leftmost = (int(extreme_points["leftmost"][0] + dx), int(extreme_points["leftmost"][1] + dy))
    rightmost = (int(extreme_points["rightmost"][0] + dx), int(extreme_points["rightmost"][1] + dy))
    topmost = (int(extreme_points["topmost"][0] + dx), int(extreme_points["topmost"][1] + dy))
    bottommost = (int(extreme_points["bottommost"][0] + dx), int(extreme_points["bottommost"][1] + dy))

    for pt in [leftmost, rightmost, topmost, bottommost]:
        cv2.circle(result_img, pt, radii["gabarit_points"], colors["gabarit_points"], -1)

    # =============================
    # === ОТРИСОВКА РАЗМЕРОВ ===
    # =============================

    # --- Размер L ---
    L_px = gabarit_data["dimensions_px"]["L_px"]
    L_mm = gabarit_data.get("dimensions_mm", {}).get("L_mm", scale_to_mm(L_px, pixels_per_mm))
    draw_dimension_universal(
        result_img, leftmost, rightmost,
        f"L = {L_mm:.2f} mm / {L_px:.0f} px",
        config, pixels_per_mm, "L",
        neck_orientation=neck_orientation
    )

    # --- Размер Z ---
    Z_px = gabarit_data["dimensions_px"]["Z_px"]
    Z_mm = gabarit_data.get("dimensions_mm", {}).get("Z_mm", scale_to_mm(Z_px, pixels_per_mm))
    draw_dimension_universal(
        result_img, topmost, bottommost,
        f"Z = {Z_mm:.2f} mm / {Z_px:.0f} px",
        config, pixels_per_mm, "Z",
        neck_orientation=neck_orientation
    )

    # --- Размер d1 ---
    if bottom_data is not None:
        bottom_points = bottom_data["bottom_points"]
        top_pt_orig = bottom_points["top_pt"]
        bottom_pt_orig = bottom_points["bottom_pt"]

        shifted_top_pt = (int(top_pt_orig[0] + dx), int(top_pt_orig[1] + dy))
        shifted_bottom_pt = (int(bottom_pt_orig[0] + dx), int(bottom_pt_orig[1] + dy))

        def clip_point(pt, w, h):
            return (max(0, min(pt[0], w - 1)), max(0, min(pt[1], h - 1)))

        h_img, w_img = result_img.shape[:2]
        clipped_top_pt = clip_point(shifted_top_pt, w_img, h_img)
        clipped_bottom_pt = clip_point(shifted_bottom_pt, w_img, h_img)

        # Рисуем точки дна
        cv2.circle(result_img, clipped_top_pt, radii["bottom_points"], colors["bottom_points"], -1)
        cv2.circle(result_img, clipped_bottom_pt, radii["bottom_points"], colors["bottom_points"], -1)

        # Определяем сторону дна на основе ориентации горлышка
        d1_side = "right" if neck_orientation == "neck_left" else "left"

        d1_config = config.copy()
        d1_config["dimensions"]["d1"]["side"] = d1_side

        d1_px = bottom_data["dimensions_px"]["d1_px"]
        d1_mm = bottom_data.get("dimensions_mm", {}).get("d1_mm", scale_to_mm(d1_px, pixels_per_mm))
        draw_dimension_universal(
            result_img, clipped_top_pt, clipped_bottom_pt,
            f"d1 = {d1_mm:.2f} mm / {d1_px:.0f} px",
            d1_config, pixels_per_mm, "d1",
            neck_orientation=neck_orientation
        )

    # --- Сохранение через единый механизм ---
    save_debug_image(result_img, "10_size_visualization", image_id, step_flag=True)