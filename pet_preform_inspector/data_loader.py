# data_loader.py
import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import config # Импортируем config для BASE_PATH


# --- Путь к входным изображениям (относительно BASE_PATH) ---
INPUT_FOLDER = config.BASE_PATH / "input_images"


def _load_single_image(image_path: str) -> np.ndarray:
    """Внутренняя функция для загрузки одного изображения из файла."""
    img = cv2.imread(str(image_path)) # str() нужен, так как cv2.imread ожидает строку
    if img is None:
        error_msg = f"Не удалось загрузить изображение: {image_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    logger.debug(f"Изображение успешно загружено: {image_path}")
    return img


def load_images_from_folder(input_folder: Path = INPUT_FOLDER):
    """
    Загружает изображения из указанной папки.
    Возвращает список кортежей: (image_id, image_data).
    image_id - уникальный идентификатор (например, имя файла).
    image_data - загруженное изображение (np.ndarray).
    """
    # Проверяем существование папки
    if not input_folder.is_dir():
        logger.error(f"Папка не существует: {input_folder}")
        return []

    image_files = [
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
    ]

    if not image_files:
        logger.warning(f"Нет изображений в папке {input_folder}")
        return []

    logger.info(f"Найдено {len(image_files)} изображений для обработки из {input_folder}")

    images = []
    for image_file_path in image_files:
        filename = image_file_path.name
        try:
            img = _load_single_image(str(image_file_path)) # str() для cv2
            images.append((filename, img))
            logger.debug(f"Изображение загружено: {filename}")
        except Exception as e:
            logger.warning(f"Ошибка при загрузке {filename}: {e}")
            continue

    return images


# Здесь можно добавить другие функции для загрузки из других источников в будущем
# def load_images_from_api(api_endpoint: str, ...):
#     ...
#     return [(image_id, img_data), ...]
