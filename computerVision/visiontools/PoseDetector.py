import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self, mode=False, maxHands=2, model_compl=1, detection_con=0.5, track_con=0.5):
        self.mpPoses = mp.solutions.pose
        self.pose =  self.mpPoses.Pose() #mode, maxHands, model_compl, detection_con, track_con)
        self.mpDraw = mp.solutions.drawing_utils

    def FindPoses(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.pose_marks = self.pose.process(imgRGB)

    def DrawPoses(self, image):
        if self.pose_marks.pose_landmarks:
                self.mpDraw.draw_landmarks(image, self.pose_marks.pose_landmarks, self.mpPoses.POSE_CONNECTIONS)
        return image