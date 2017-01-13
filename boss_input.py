# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        # print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if  file_or_dir.startswith('images') or file_or_dir.endswith('.jpg') or  file_or_dir.endswith('.pgm') or file_or_dir.endswith('.png'):
                
                print(abs_path + " >> " + path[:5])

                image = read_image(abs_path)
                
                if image != None:
                    images.append(image)
                    labels.append(path)

    return images, labels

import cv2
cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

def extract_face(img):
    frame = img#cap.read()
    # グレースケール変換
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # カスケード分類器の特徴量を取得する
    # 物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
    #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
    if len(facerect) > 0:
        print('face detected')
        color = (255, 255, 255)  # 白
        for rect in facerect:
#                 clear_output()
            # 検出した顔を囲む矩形の作成
            #cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

            x, y = rect[0:2]
            width, height = rect[2:4]
            image = frame[y - 10: y + height, x: x + width]
            return image

def read_image(file_path):
    image = cv2.imread(file_path)
    image = extract_face(image)
    if image == None:
        return None
    
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('boss') else 1 for label in labels])

    return images, labels
