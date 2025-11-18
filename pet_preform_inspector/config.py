# config.py
import os
from pathlib import Path

# --- Базовый путь проекта ---
BASE_PATH = Path(__file__).resolve().parent  # Путь к папке, где лежит config.py (корень проекта)

# --- Пути (относительно BASE_PATH) ---
OUTPUT_FOLDER = BASE_PATH / "output_results"
LOGS_FOLDER = BASE_PATH / "logs"
DEBUG_OUTPUT_FOLDER = BASE_PATH / "debug_output"

# --- Параметры обработки ---
MIN_CONTOUR_AREA = 200
EPSILON_FACTOR = 0.01
BINARY_METHOD = "CANNY"  # "CANNY", "OTSU", "COLOR_HSV", "BACKGROUND_SUB", BACKGROUND_SUB_CANNY, DISTANCE_TRANSFORM или "ADAPTIVE"

# --- Параметры предварительной обработки и бинаризации ---
# Gaussian Blur
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)  # Должно быть нечетное число, например, (3, 3), (5, 5), (7, 7)
GAUSSIAN_BLUR_SIGMA_X = 0          # 0 означает, что sigma вычисляется из размера ядра

# Canny Edge Detection (используется только если BINARY_METHOD == "CANNY")
CANNY_THRESHOLD1 = 30             # Первый порог для гистерезиса 10
CANNY_THRESHOLD2 = 110          # Второй порог для гистерезиса 50

# Морфологические операции (для закрытия/открытия)
#MORPHOLOGY_KERNEL_SIZE = 7         # Размер квадратного ядра для закрытия/открытия в _binarize_canny, apply_morphology
OTSU_MORPHOLOGY_KERNEL_SIZE = 5    # Размер квадратного ядра для закрытия в _binarize_otsu

MORPHOLOGY_KERNEL_SIZE = 9  # или 5 — зависит от разрешения

# --- Флаги отладки для визуализации на разных шагах ---
STEP_1_DETECT_CONTOUR = True    # Визуализация в preprocessing.contour_detector
STEP_2_ALIGN_CONTOUR = True       # Визуализация в analyze (visualize_axis_with_centroid)
STEP_3_MEASURE_DIMENSIONS = True  # Визуализация в measurement.geometry (visualize_neck_bottom_zones)
STEP_4_CHECK_DEFECTS = True       # Визуализация в measurement.defect_checker (check_straightness)
STEP_5_SAVE_RESULT = True         # Визуализация в utils (save_result_image)
STEP_10_STRAIGHTNESS = False       # Визуализация прямолинейности
