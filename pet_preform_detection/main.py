import os
from config import METHOD_DETECT
from detect_cv import objects_detection_cv2
from detect_yolo import objects_detection_yolo

def main():
    if METHOD_DETECT=="YOLO":
        objects_detection_yolo();
    else: objects_detection_cv2();

if __name__ == "__main__":
    main()
