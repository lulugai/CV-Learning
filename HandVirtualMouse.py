import cv2
import numpy as np
import HandTrackModule as htm
import autopy

brushthickness = 15

def main():
    cap = cv2.VideoCapture(0)
    wCam, hCam = 1280, 720
    cap.set(3, wCam)
    cap.set(4, hCam)
    drawcolor = (255,0,0)
    xp, yp = 0, 0
    imgcanvas = np.zeros((hCam, wCam, 3), np.uint8)

    detector = htm.HandDetector(min_tracking_confidence=0.7)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findhands(img)    
        Lmlist = detector.findpositions(img, draw=False)
        if len(Lmlist) != 0:
            x1, y1 = Lmlist[8][1:] 
            x2, y2 = Lmlist[12][1:]
            # check which fingers are up
            fingers = detector.fingersUp()
            # if selection mode - two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                drawcolor = (0,0,255)
                cv2.rectangle(img, (x1, y1 - 35), (x2, y2 + 35), drawcolor, cv2.FILLED)
                # print('selection mode')

            # if drawing mode - index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, (0,0,255), cv2.FILLED)
                # print('draw mode')
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
                xp, yp = x1, y1

        imggray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imggray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgcanvas)
        cv2.imshow('Image', img)
        # cv2.imshow('Canvas', imgInv)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()