import os, sys
from PIL import Image
import cv2

from visiontools.HandDetector import HandTracker



tracker = HandTracker(maxHands=1)
dir_path_images = os.path.join('Signlanguage', 'Data','images')

#converting the image files from png jpeg to png
for file in os.listdir(dir_path_images):
    handle = os.path.join(dir_path_images, file)
    if '.jpeg' in handle:
        image_jpeg = Image.open(handle)
        image_jpeg.save(os.path.join(dir_path_images, file.replace('jpeg','png')))
        os.remove(handle)

#writing coordingate files
for file in os.listdir(dir_path_images):
    handle = os.path.join(dir_path_images, file)
    image = cv2.imread(handle)
    tracker.FindHands(image)
    tracker.FindPositions(image)
    tracker.hand_coords = tracker.CenterHand(tracker.hand_coords, 5)
    tracker.WriteToFile(os.path.join('Signlanguage', 'Data','coordinates',file.replace('png','txt')))

    #testing the results of the mashine model
    while True:
        image = tracker.DrawHands(image)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == ord('q'): break