# data_loader.py
import os
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import config # Импортируем config для BASE_PATH

import sys
import tempfile
import shutil
from PIL import Image
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Union

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)

# --- Путь к входным изображениям (относительно BASE_PATH) ---
INPUT_FOLDER = config.BASE_PATH / "input_images"

# =============================================
# 1. ИНИЦИАЛИЗАЦИЯ И ПРОВЕРКА ОС
# =============================================

def get_platform_info() -> Dict[str, Any]:
    """Возвращает информацию о платформе"""
    platform_info: Dict[str, Any] = {
        'is_windows': sys.platform == 'win32',
        'is_linux': sys.platform.startswith('linux'),
        'is_mac': sys.platform == 'darwin',
        'platform': sys.platform
    }
    logger.info(f"Платформа: {platform_info}")
    return platform_info


# =============================================
# 2. ВАРИАНТ ДЛЯ WINDOWS ЧЕРЕЗ СИСТЕМНЫЕ ВЫЗОВЫ
# =============================================

# Глобальные переменные для Windows API
windows_api: Optional[Dict[str, Any]] = None
ctypes = None
wintypes = None


def _init_windows_api() -> Optional[Dict[str, Any]]:
    """Инициализация Windows API"""
    global windows_api, ctypes, wintypes

    if sys.platform != 'win32':
        return None

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # Константы Windows
        GENERIC_READ = 0x80000000
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        # Настройка функций
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
        ]

        kernel32.GetFileSize.restype = wintypes.DWORD
        kernel32.GetFileSize.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
        ]

        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID
        ]

        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        windows_api = {
            'kernel32': kernel32,
            'constants': {
                'GENERIC_READ': GENERIC_READ,
                'OPEN_EXISTING': OPEN_EXISTING,
                'FILE_ATTRIBUTE_NORMAL': FILE_ATTRIBUTE_NORMAL,
                'INVALID_HANDLE_VALUE': INVALID_HANDLE_VALUE
            }
        }
        return windows_api
    except Exception as e:
        logger.warning(f"Не удалось инициализировать Windows API: {e}")
        return None


# Инициализируем Windows API при загрузке модуля
_init_windows_api()


def _load_image_windows(image_path: str) -> np.ndarray:
    """Загрузка изображения через Windows API"""
    global windows_api, ctypes, wintypes

    if not windows_api:
        raise RuntimeError("Windows API не доступно")

    kernel32: ctypes.WinDLL = windows_api['kernel32']
    constants: Dict[str, Any] = windows_api['constants']

    # Открываем файл
    hfile: int = kernel32.CreateFileW(
        image_path,
        constants['GENERIC_READ'],
        0, None, constants['OPEN_EXISTING'],
        constants['FILE_ATTRIBUTE_NORMAL'], None
    )

    if hfile == constants['INVALID_HANDLE_VALUE']:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    try:
        # Получаем размер файла
        file_size_high = wintypes.DWORD()
        file_size_low = kernel32.GetFileSize(hfile, ctypes.byref(file_size_high))
        file_size = (file_size_high.value << 32) + file_size_low

        if file_size == 0:
            raise ValueError("Файл пустой")

        # Читаем файл
        buffer: ctypes.Array[ctypes.c_char] = ctypes.create_string_buffer(file_size)
        bytes_read = wintypes.DWORD()

        success: bool = kernel32.ReadFile(
            hfile, buffer, file_size, ctypes.byref(bytes_read), None
        )

        if not success or bytes_read.value != file_size:
            raise IOError(f"Ошибка чтения файла: {image_path}")

        # Декодируем изображение
        np_buffer: np.ndarray = np.frombuffer(buffer.raw, dtype=np.uint8, count=file_size)
        img: np.ndarray = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Не удалось декодировать изображение")

        logger.debug(f"Изображение загружено через Windows API: {image_path}")
        return img

    finally:
        kernel32.CloseHandle(hfile)


# =============================================
# 3. ВАРИАНТ ДЛЯ LINUX
# =============================================

def _load_image_linux(image_path: str) -> np.ndarray:
    """Загрузка изображения на Linux"""
    # На Linux обычно нет проблем с кириллицей, используем стандартный метод
    img: Optional[np.ndarray] = cv2.imread(image_path)

    if img is not None:
        logger.debug(f"Изображение загружено стандартным методом: {image_path}")
        return img

    # Если стандартный метод не сработал, пробуем Pillow
    return _load_image_pillow(image_path)


# =============================================
# УНИВЕРСАЛЬНЫЕ МЕТОДЫ (ДЛЯ ВСЕХ ОС)
# =============================================

def _load_image_pillow(image_path: str) -> np.ndarray:
    """Загрузка через Pillow (кроссплатформенный)"""
    try:
        pil_image: Image.Image = Image.open(image_path)
        img_array: np.ndarray = np.array(pil_image)

        if pil_image.mode == 'RGB':
            img: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'RGBA':
            img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pil_image.mode == 'L':
            img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img = img_array

        logger.debug(f"Изображение загружено через Pillow: {image_path}")
        return img
    except Exception as e:
        raise ValueError(f"Pillow не смог загрузить изображение: {e}")


def _load_image_temp_file(image_path: str) -> np.ndarray:
    """Загрузка через временный файл (кроссплатформенный)"""
    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name

        shutil.copy2(image_path, temp_filename)
        img: Optional[np.ndarray] = cv2.imread(temp_filename)

        if img is None:
            raise ValueError("Не удалось загрузить через временный файл")

        logger.debug(f"Изображение загружено через временный файл: {image_path}")
        return img
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass


# =============================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================

def _load_single_image(image_path: str) -> np.ndarray:
    """Основная функция загрузки изображения"""
    platform_info: Dict[str, Any] = get_platform_info()

    # Проверяем существование файла
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не существует: {image_path}")

    methods_to_try: List[Tuple[str, Callable[[str], np.ndarray]]] = []

    # Выбираем методы в зависимости от ОС
    if platform_info['is_windows']:
        methods_to_try = [
            ("Windows API", _load_image_windows),
            ("Pillow", _load_image_pillow),
            ("Временный файл", _load_image_temp_file)
        ]
    else:  # Linux, Mac и другие
        def standard_method(path: str) -> np.ndarray:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Стандартный метод вернул None")
            return img

        methods_to_try = [
            ("Стандартный метод", standard_method),
            ("Pillow", _load_image_pillow),
            ("Временный файл", _load_image_temp_file)
        ]

    # Пробуем методы по порядку
    last_error: Optional[Exception] = None
    for method_name, method_func in methods_to_try:
        try:
            img = method_func(image_path)
            logger.debug(f"Изображение загружено {method_name}: {image_path}")
            return img
        except Exception as e:
            last_error = e
            logger.warning(f"Метод {method_name} не сработал: {e}")
            continue

    # Если все методы не сработали
    error_msg = f"Все методы загрузки не удались для: {image_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg) from last_error

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

# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

if __name__ == "__main__":
    # Тестируем функцию
    logging.basicConfig(level=logging.DEBUG)

    test_image: str = "test_image.jpg"  # Замените на реальный путь

    try:
        img: np.ndarray = _load_single_image(test_image)
        print(f"✅ Изображение успешно загружено! Размер: {img.shape}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")


