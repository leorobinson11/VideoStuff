import os, sys
import numpy as np
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, maxHands=2, model_compl=1, detection_con=0.5, track_con=0.5):
        #setting up conditions for mediapipe mashine model
        self.mpHands = mp.solutions.hands
        self.hands =  self.mpHands.Hands(mode, maxHands, model_compl, detection_con, track_con)
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, image):
        #running the mediapipe model on an image
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.hand_marks = self.hands.process(imgRGB)

    def FindPositions(self, image, Handnumber=0, inc_z=False):
        #iterating through the landmark positions aswell as converting them from fractions to pixels
        h, w, c = image.shape
        self.hand_coords = []
        if self.hand_marks.multi_hand_landmarks:
            myHand = self.hand_marks.multi_hand_landmarks[Handnumber]
            for lm in myHand.landmark:
                coords = [lm.x * w, lm.y * h]
                if inc_z: coords += [lm.z * c]
                self.hand_coords.append(coords)
        return self.hand_coords

    def WriteToFile(self, file_path):
        #writing the current landmark coordinates to a file
        with open(file_path, 'w') as file:
            for coords in self.hand_coords:
                str_form = ''.join(map(lambda l:str(l)+' ', coords)) + '\n'
                file.write(str_form)

    def ReadFromFile(self, file_path):
        #reading any landmark positions to a file
        with open(file_path) as file:
            for line in file:
                stripped_line = line.strip()
                yield list(map(float, stripped_line.split()))
        
    def ReadAllFromDirectory(self, directory_path):
        #creating a dirctionary of landmark coordinates from a directory refrence files
        self.hand_positions_saved = {}
        for file in os.listdir(directory_path):
            handle = os.path.join(directory_path, file)
            self.hand_positions_saved.update({
                file.replace('.txt', ''):list(self.ReadFromFile(handle))
            })

    def HandCenter(self, *args):
        #Finding the geometrik center of the hand (used for unexact comparisons)
        center = [0,0,0]
        for hand_cords in args:
            for coords in hand_cords:
                center = np.add(center, coords)
        return np.divide(center, sum(map(len, args)))

    def CenterHand(self,hand_coords, reference_landmark):
        #centering the hand so that some landmark chosen as the origin equals zero
        ref = hand_coords[reference_landmark]
        return list(map(
            np.subtract, hand_coords, [ref for _ in range(len(hand_coords))]
        ))

    def MeanDistance(self, hand_coords1, hand_coords2):
        #Calculating the avarage distance between the landmarks of two hand coordinate lists
        total = 0
        for (coords1, coords2) in zip(hand_coords1, hand_coords2):
            total += np.linalg.norm(coords2-coords1)
        return np.divide(total, 2)

    def Angle(self):
        ...
    
    def Rotate(self):
        ...

    def DrawHands(self, image):
        #Drawing the Hand landmarks aswell as the connections onto an image
        if self.hand_marks.multi_hand_landmarks:
            for handLms in self.hand_marks.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image