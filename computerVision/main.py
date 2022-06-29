import cv2

from visiontools.HandDetector import HandTracker
from visiontools.PoseDetector import PoseTracker

def Hand():
    cap = cv2.VideoCapture(0)
    detector = HandTracker(detection_con=0.3,track_con=0.3)
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        detector.FindHands(img)
        img = detector.DrawHands(img)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'): break
    cap.release()
    cv2.destroyAllWindows

if __name__ == '__main__':
    import OpenClosed
    #Hand()