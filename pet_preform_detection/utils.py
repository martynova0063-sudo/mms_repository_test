import os
import cv2
from config import PATH_VIDEO, FILE_FORMAT, OFFSET_CONTOUR, PATCH_IMAGE_SAVE

def save_box(ROI_number, i, original):
  """
  Функция сохранени изображения
  """
  cv2.imwrite(f"{PATCH_IMAGE_SAVE}ROI_{ROI_number}_i_{i}.{FILE_FORMAT}", original)
  
def save_frame(i, frame):
  """
  Функция сохранени фрейма
  """
  cv2.imwrite(f"images/frame/frame_{i}.{FILE_FORMAT}", frame)


def cropped_image(ROI_number, i, original, x, y, x2, y2):
  """
  Увеличивает рамку на коэффициент, 
  поворачивает изображение если высота превышает ширину

  Возвращает:
  новое изображение, без сохранения в файл
  """
  x=x-OFFSET_CONTOUR
  y=y-OFFSET_CONTOUR
  x2=x2+OFFSET_CONTOUR
  y2=y2+OFFSET_CONTOUR
  ROI = original[y:y2, x:x2]
  if x2-x<y2-y: ROI =cv2.rotate(ROI, cv2.ROTATE_90_CLOCKWISE)
  return ROI

def crop_15_percent(image):   
    """
    Увеличивает рамку на коэффициент, 
    поворачивает изображение если высота превышает ширину

    Возвращает:
    новое изображение, без сохранения в файл
  """ 
    # Получаем размеры изображения
    height, width = image.shape[:2]    
    # Вычисляем 15% от размеров
    crop_height = int(height * 0.20)
    crop_width = int(width * 0.05)    
    # Вырезаем изображение
    cropped_image = image[
        crop_height:height-crop_height, 
        crop_width:width-crop_width
    ]    
    return cropped_image

def read_video():
  """
  Функция чтения видео

  Возвращает:
  Количество кадров и прочитанное видео
  """
  cap = cv2.VideoCapture(PATH_VIDEO)
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  return cap, length

