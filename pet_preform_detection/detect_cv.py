import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils import save_box, read_video, save_frame, cropped_image, crop_15_percent
from rotate_position import find_contours, align_contour, rotate_coordinates
import os
from config import (INPUT_HEIGHT, INPUT_WIDTH, FILE_FORMAT, PATCH_IMAGE_SAVE)

def objects_detection_cv2():
    cap, length = read_video()
    # Цвета для визуализации
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    ROI_number = 0
    for i in range(length):
        # Чтение кадра с камеры
        ret, frame = cap.read()
        # Получение размеров кадра
        (H, W) = frame.shape[:2]
        # Предварительная обработка изображения
        # Изменение размера и нормализация
        image = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        # Обработка предсказаний
        if 1==1:
            # Поиск контуров всех объектов на фрейме
            contours=find_contours(frame)             
            for contour in contours:
            # Фильтрация по площади
                if 600<cv2.contourArea(contour)<2800:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 125<y<260 and (20<x<220 or 270<x<550):
                        ROI =cropped_image(ROI_number, i, frame.copy(), x, y, x+w, y+h)
                        new_contour=find_contours(ROI)  
                        max_contour=max(new_contour, key=cv2.contourArea)
                        rotated_image, angle=align_contour(max_contour, ROI)
                        save_box(ROI_number, i, crop_15_percent(rotated_image))
                        #cv2.imwrite(f"{PATCH_IMAGE_SAVE}rotated_{ROI_number}_i_{i}.{FILE_FORMAT}", crop_15_percent(rotated_image))
                    ROI_number += 1
            # Отрисовка контуров
            for contour in contours:
            # Фильтрация по площади
                if 600<cv2.contourArea(contour)<2800:
                     x, y, w, h = cv2.boundingRect(contour)
                     if 125<y<260 and (20<x<220 or 270<x<550):
                      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                     # Вывод информации о позиции
                     cv2.putText(frame, f"X: {x}, Y: {y}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
                     M = cv2.moments(contour)
                     cx = int(M['m10'] / M['m00'])
                     cy = int(M['m01'] / M['m00'])
                     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            save_frame(i, frame)
    # Освобождение ресурсов
    cap.release()
