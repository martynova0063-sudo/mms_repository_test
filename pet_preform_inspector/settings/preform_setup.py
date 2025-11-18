# settings/preform_setup.py
import json
from pathlib import Path
from loguru import logger
import config # Импортируем config, чтобы получить BASE_PATH


# --- Настройки для текущего запуска ---
# Тип преформы
PREFORM_TYPE = "type_A"  # Должен соответствовать ключу в JSON

# --- Путь к файлу с описаниями преформ (относительно BASE_PATH) ---
PREFORM_CONFIGS_PATH = config.BASE_PATH / "preform_configs.json"

# --- Коэффициент калибровки для пересчёта расстояния из пикселей в миллиметры---
CALIBRATION_PPM = 8.0


def get_active_dimensions():
    """
    Загружает и возвращает словарь с параметрами размеров
    для активного типа преформы (PREFORM_TYPE) из JSON файла.
    """
    # Загружаем файл при каждом вызове
    if not PREFORM_CONFIGS_PATH.exists():
        logger.error(f"Файл конфигураций преформ не найден: {PREFORM_CONFIGS_PATH}")
        return {}

    try:
        with open(PREFORM_CONFIGS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        configs = data.get("preform_types", {})
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в файле {PREFORM_CONFIGS_PATH}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигураций преформ: {e}")
        return {}

    # Ищем конфигурацию для активного типа
    active_config = configs.get(PREFORM_TYPE)

    if not active_config:
        logger.error(f"Тип преформы '{PREFORM_TYPE}' не найден в файле конфигураций '{PREFORM_CONFIGS_PATH}'.")
        return {}

    dimensions = active_config.get("dimensions", {})
    logger.info(f"Загружены параметры размеров для типа '{PREFORM_TYPE}' из {PREFORM_CONFIGS_PATH}.")
    return dimensions
