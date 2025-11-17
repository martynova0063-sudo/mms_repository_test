import cv2
import numpy as np
import math

def find_contours(img): 
    """
    Функция поиска контура
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #бинаризация (0 - порогового преобразования) изображения
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Морфологическая обработка для очистки эрозия
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #Удаляет мелкие объекты, которые меньше структурного элемента Помогает избавиться от шума эрозия ->дилатация
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #дилатация->эрозия Закрывает небольшие отверстия внутри объектов
    #Соединяет близко расположенные объекты iterations=2 означает двукратное применение операции
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Поиск контуров метод аппроксимации контуров cv2.CHAIN_APPROX_SIMPLE — сохраняются только конечные точки прямых линий
    # cv2.RETR_EXTERNAL - поиск только внешних контуров
    #входное изображение. Должно быть бинарным (обычно результат порогового преобразования или морфологических операций)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def align_contour(cnt, image):
    """
    Вычисляет угол поворота, относительно главной оси

    Возвращает:
    угол поворта и изменённое изображение
    """
    # Вычисляем моменты контура
    moments = cv2.moments(cnt) 
    # Вычисляем угол поворота
    if moments['mu02'] == moments['mu20']:
        angle = 0
    else:
        angle = 0.5 * math.atan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        angle = np.degrees(angle)    
    # Корректируем угол
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = angle    
    # Получаем размеры изображения
    height, width = image.shape[:2]    
    # Создаем матрицу поворота
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)    
    # Применяем поворот
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return rotated_image, angle

def rotate_coordinates(x, y, width, height, angle, center=None):
    """
    Вычесляет новые координаты, исходя из угла поворота

    Возвращает:
        x, y верхнего угла и ширина и высота
    """
    # Преобразуем угол в радианы
    angle_rad = np.radians(angle)    
    # Если центр не задан, используем центр изображения
    if center is None:
        center = (width / 2, height / 2)    
    # Распаковываем координаты центра
    cx, cy = center    
    # Создаем матрицу поворота
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)    
    # Пересчитываем координаты верхнего левого угла
    x1 = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
    y1 = cy + (x - cx) * sin_theta + (y - cy) * cos_theta    
    # Пересчитываем координаты нижнего правого угла
    x2 = cx + (x + width - cx) * cos_theta - (y + height - cy) * sin_theta
    y2 = cy + (x + width - cx) * sin_theta + (y + height - cy) * cos_theta    
    # Вычисляем новые ширину и высоту
    new_width = abs(x2 - x1)
    new_height = abs(y2 - y1)    
    if x1<0: x1=0 
    else: int(x1)
    if y1<0: y1=0 
    else: int (y1) 
    return (int(x1), int(y1), int(new_width), int(new_height))
