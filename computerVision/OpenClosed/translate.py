import os

import cv2
import asyncio
from visiontools.HandDetector import HandTracker



async def Translate(image):
    result = 'open'
    detector = HandTracker(maxHands=1)
    detector.ReadAllFromDirectory(os.path.join('Signlanguage', 'Data','coordinates'))
    detector.FindHands(image)
    detector.FindPositions(image)
    if len(detector.hand_coords) > 0:
        centered_current_hand = detector.CenterHand(detector.hand_coords, 5)
        for (letter, hand) in detector.hand_positions_saved.items():
            if detector.MeanDistance(centered_current_hand, hand) < 2000:
                result = 'closed'
    return result, detector.DrawHands(image)

async def HandCam():
    cap = cv2.VideoCapture(0)
    while True:
        #reading image
        success, image = cap.read()
        image = cv2.flip(image,1)

        #processing image
        result, image =  await asyncio.create_task(Translate(image))

        #outputting image
        image = cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == ord('q'): break
    cap.release()
    cv2.destroyAllWindows