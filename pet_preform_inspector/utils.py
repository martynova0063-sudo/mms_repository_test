# utils.py
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import config


def setup_logging():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = config.LOGS_FOLDER / f"inspection_{timestamp}.log"

    logger.remove()
    config.LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file_path), rotation="10 MB", retention="7 days", level="INFO", encoding="utf-8")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    return logger


def point_to_line_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return numerator / denominator if denominator > 0 else 0.0


def scale_to_mm(value_in_pixels, pixels_per_mm):
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    return float(value_in_pixels) / float(pixels_per_mm)


def save_result_image(img, contours, approx, defects, output_path):
    img_with_contour = img.copy()
    cv2.drawContours(img_with_contour, [approx], -1, (0, 255, 0), 2)

    y_offset = 30
    for i, (name, val, ok) in enumerate(defects):
        color = (0, 255, 0) if ok else (0, 0, 255)
        if name == "protrusions":
            text_val = "YES" if val else "NO"
            status = "OK" if ok else "BRAK"
            text = f"{name}: {text_val} ({status})"
        else:
            status = "OK" if ok else "BRAK"
            text = f"{name}: {val:.2f} mm ({status})"
        cv2.putText(
            img_with_contour,
            text,
            (10, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_with_contour)
    logger.info(f"Результат сохранён: {output_path}")


def create_output_dir():
    config.OUTPUT_FOLDER.mkdir(exist_ok=True)
    config.LOGS_FOLDER.mkdir(exist_ok=True)
    (config.BASE_PATH / "debug_output").mkdir(parents=True, exist_ok=True)


def save_debug_image(img, step_name, image_id=None, step_flag: bool = False): # Добавляем step_flag
    """
    Сохраняет отладочное изображение, если соответствующий флаг включён.
    Имя файла формируется как: <image_id>_<step_name>.jpg
    Если image_id не передан, используется временная метка.
    """
    if not step_flag: # Проверяем переданный флаг
        return

    debug_dir = config.BASE_PATH / "debug_output"
    debug_dir.mkdir(exist_ok=True)

    if image_id is None:
        # Если image_id не передан, используем временную метку
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"debug_{timestamp}_{step_name}.jpg"
    else:
        # Используем original image_id для имени файла
        # Очищаем image_id от недопустимых символов
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(image_id))
        filename = f"{safe_id}_{step_name}.jpg"

    output_path = debug_dir / filename

    if len(img.shape) == 2:
        out_img = img
    elif len(img.shape) == 3 and img.shape[2] in (1, 3):
        out_img = img
    else:
        raise ValueError(f"Неподдерживаемая форма изображения: {img.shape}")

    cv2.imwrite(str(output_path), out_img)
    logger.debug(f"Отладка: сохранено {output_path}")
