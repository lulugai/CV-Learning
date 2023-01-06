import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime, cTime = 0, 0
while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgrgb)
    h, w, c = img.shape
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, 
                                    drawSpec, drawSpec)
            for id, Lm in enumerate(faceLms.landmark):
                x, y = int(Lm.x*w), int(Lm.y*h)
                # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.8, 
                #             (0, 255, 0), 1)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {str(int(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, 
                (0, 255, 0), 2)
    cv2.namedWindow('Image', 0)
    cv2.resizeWindow('Image', 1000, 800)
    cv2.imshow('Image', img)
    cv2.waitKey(1)






