import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils import save_box, read_video, save_frame, cropped_image, crop_15_percent
from rotate_position import find_contours, align_contour, rotate_coordinates
from config import CONFIDENCE_THRESHOLD, PATCH_IMAGE_SAVE, FILE_FORMAT

def detect_objects(img, model, ROI_number):
    # Настройка параметров предсказания
    results = model.predict(
      img,
      conf=CONFIDENCE_THRESHOLD,  # Минимальная уверенность
      iou=0.45,   # Порог пересечения
      max_det=1000  # Максимальное количество обнаружений
    )
    i=0
    # Обработка результатов
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Извлечение координат bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Извлечение класса и уверенности
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Получение названия класса
            class_name = model.names[cls]

            #Сохранение прямоугольника
            ROI =cropped_image(ROI_number, i, img.copy(), int(x1), int(y1), int(x2), int(y2))
            new_contour=find_contours(ROI)  
            max_contour=max(new_contour, key=cv2.contourArea)
            rotated_image, angle=align_contour(max_contour, ROI) 
            save_box(ROI_number, i, crop_15_percent(rotated_image))
            i+= 1 #Счетчик изображений

            # Рисование bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Добавление метки класса
            cv2.putText(img, f'{class_name} {conf:.2f}', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

    return img

def objects_detection_yolo():
    # Загрузка обученной модели
    model = YOLO('custom_yolo/model/best.pt')
    # чтение видеофайла
    cap, length = read_video()
    ROI_number = 0
    for i in range(length):
        ret, frame = cap.read()
        if not ret: break
        # Получение размеров кадра
        (H, W) = frame.shape[:2]
        detect_objects(frame, model, ROI_number)
        save_frame(i, frame)
        #Счетчик фреймов
        ROI_number += 1
    # Освобождение ресурсов
    cap.release()

