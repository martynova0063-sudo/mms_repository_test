# Параметры для обработки
METHOD_DETECT= "CV2"  # "CV2" или "YOLO"

# минимальная уверенность - точность предсказания модели для детекции (необязательно)
CONFIDENCE_THRESHOLD = 0.5

#размер фрейма
INPUT_WIDTH = 640
INPUT_HEIGHT = 352
#путь к видеофайлу
PATH_VIDEO = 'video/track001.mp4'

#формат выгружаемых файлов jpg, png, bmp, tiff
FILE_FORMAT ="jpg"

#смещение контура прямоугольник в px
OFFSET_CONTOUR=10

#путь для сохранения картинок
PATCH_IMAGE_SAVE="images/box/"

