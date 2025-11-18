# Программа формирования изображений с разметкой по размеченному датасету.
# Размеченный датасет был выгружен из Label Studio в формате YOLO with images
# Т.е есть папки с исходными изображениями, разметкой и классами

import cv2
import os
from PIL import Image
import numpy as np

# Пути к папкам с данными YOLO (с кириллицей)
images_dir = r"D:\ISZF\Университет Искусственного Интеллекта\Стажировки\АМАИ\DATASET\yolo\images"
labels_dir = r"D:\ISZF\Университет Искусственного Интеллекта\Стажировки\АМАИ\DATASET\yolo\labels"
classes_file = r"D:\ISZF\Университет Искусственного Интеллекта\Стажировки\АМАИ\DATASET\yolo\classes.txt"
output_dir = r"D:\ISZF\Университет Искусственного Интеллекта\Стажировки\АМАИ\DATASET\yolo\marked_images"

# Создаём папку для результатов
os.makedirs(output_dir, exist_ok=True)

# Загружаем классы
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# Получаем список всех изображений
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]

# Проходим по всем изображениям
for image_file in image_files:
    image_path = os.path.normpath(os.path.join(images_dir, image_file))
    label_file = os.path.normpath(os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt'))

    # Загружаем изображение с помощью PIL
    try:
        pil_image = Image.open(image_path)
        # Конвертируем в формат OpenCV
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Ошибка при открытии {image_path}: {e}")
        continue

    # Загружаем разметку YOLO
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # Получаем координаты рамки
            img_height, img_width = image.shape[:2]
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # # Подписываем класс
            # class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
            # cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Конвертируем обратно в формат PIL для сохранения
    result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Сохраняем размеченное изображение в формате JPG
    output_filename = f"marked_{os.path.splitext(image_file)[0]}.jpg"
    output_path = os.path.normpath(os.path.join(output_dir, output_filename))

    try:
        result_image.save(output_path)
        print(f"Успешно сохранено: {output_path}")
    except Exception as e:
        print(f"Не удалось сохранить: {output_path}, ошибка: {e}")
