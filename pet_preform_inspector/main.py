# main.py
import os
from loguru import logger
import config
from analysis.analyzer import PreformGeometryAnalyzer
from data_loader import load_images_from_folder, INPUT_FOLDER
from utils import setup_logging, create_output_dir
from settings.preform_setup import get_active_dimensions, CALIBRATION_PPM, PREFORM_TYPE


def main():
    setup_logging()
    create_output_dir()

    logger.remove()  # удаляем стандартный handler
    logger.add(lambda msg: print(msg, end=""), level="DEBUG")  # выводим всё, включая DEBUG

    logger.info("Уровень логгирования: DEBUG включён")

    dimensions_data = get_active_dimensions()
    if not dimensions_data:
        logger.critical(f"Не удалось загрузить параметры размеров для типа '{PREFORM_TYPE}'. Завершение работы.")
        return

    current_ppm = CALIBRATION_PPM
    if current_ppm <= 0:
         logger.critical("Коэффициент калибровки pixels_per_mm недействителен. Завершение работы.")
         return

    # Инициализируем анализатор с новыми параметрами
    analyzer = PreformGeometryAnalyzer(
        logger=logger,
        pixels_per_mm=current_ppm,
        dimensions_data=dimensions_data
    )



    images = load_images_from_folder(INPUT_FOLDER)

    if not images:
        logger.warning("Нет изображений для обработки.")
        return

    logger.info(f"Начинаем обработку {len(images)} изображений с типом преформы '{PREFORM_TYPE}' и калибровкой {current_ppm} px/mm.")

    for image_id, img in images:
        logger.info(f"Обработка: {image_id}")
        try:
            verdict, measurements, defects = analyzer.analyze(img, image_id=image_id)
            logger.info(f"Результат для {image_id}: {verdict}")
        except Exception as e:
            logger.exception(f"Ошибка при анализе {image_id}: {e}")
            continue

    logger.info("✅ Все изображения обработаны.")


if __name__ == "__main__":
    main()
