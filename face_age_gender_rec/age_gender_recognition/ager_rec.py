import dlib
import tarfile
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import face_utils
from deepface import DeepFace


# Define our imshow function
def imshow(title="Image", image=None, size=6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def get_gender_and_age(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    import pprint

    img_path = path

    obj = DeepFace.analyze(img_path=img_path,
                           actions=['age', 'gender'])
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(obj)

    return obj["age"], obj["gender"]


# print(get_gender_and_age('./friends/rachelle.jpg'))
