import math
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
        self.tipids = [4, 8, 12, 16, 20]

    def findhands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img

    def findpositions(self, img, handnum=0, draw=True):
        self.Lmlist = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handnum]
            for id, lm in enumerate(myhand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.Lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        return self.Lmlist, None

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.Lmlist[p1][1:]
        x2, y2 = self.Lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = (255, 0, 255)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color, t)
            cv2.circle(img, (x1, y1), r, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), r, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), r, color, cv2.FILLED)
        length = math.hypot(x1 - x2, y1 - y2)
        
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        fingers = []
        #Thumb
        if self.Lmlist[self.tipids[0]][1] > self.Lmlist[self.tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            if self.Lmlist[self.tipids[id]][2] < self.Lmlist[self.tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

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
        Lmlist, _ = detector.findpositions(img, draw=False)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {str(int(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()