import cv2
import mediapipe as mp
import time
import numpy as np
import math
import HandTrackModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    cap = cv2.VideoCapture(0)
    pTime, cTime = 0, 0
    wCam, hCam = 1000, 800
    cap.set(3, wCam)
    cap.set(4, hCam)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volrange = volume.GetVolumeRange()
    minvol, maxvol = volrange[0], volrange[1]

    detector = htm.HandDetector(min_tracking_confidence=0.7)
    while True:
        success, img = cap.read()
        img = detector.findhands(img)    
        Lmlist = detector.findpositions(img, draw=False)
        if len(Lmlist) != 0:
            x1, y1, x2, y2 = Lmlist[4][1], Lmlist[4][2], Lmlist[8][1], Lmlist[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1, y1), 15, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)

            length = math.hypot((x1 - x2), (y1 - y2))
            # print(length)
            #length: 10 - 200
            vol = np.interp(length, [10, 200], [minvol, maxvol])
            # print(length, vol)
            volume.SetMasterVolumeLevel(vol, None)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {str(int(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()