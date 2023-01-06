import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=static_image_mode,
                                        max_num_hands=max_num_hands,
                                        model_complexity=model_complexity,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img
    def findpositions(self, img, handnum=0, draw=True):
        Lmlist = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handnum]
            for id, lm in enumerate(myhand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                Lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        return Lmlist

def main():
    cap = cv2.VideoCapture(0)
    pTime, cTime = 0, 0
    wCam, hCam = 1000, 800
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)    
        Lmlist = detector.findpositions(img, draw=False)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {str(int(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()