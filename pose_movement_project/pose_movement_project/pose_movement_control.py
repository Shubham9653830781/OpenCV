
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

cap = cv2.VideoCapture(0)

resolutions = [(640, 480), (768, 576), (1280, 720)]
cap.set(3, resolutions[1][0])
cap.set(4, resolutions[1][1])

X = [537, 464, 341, 285, 236, 188, 153, 141]
y = [50, 75, 100, 125, 150, 200, 250, 300]
coff = np.polyfit(X, y, 2)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, resolutions[1])
    img = cv2.flip(img, 1)

    vx, vy, vz = 0, 0, 0

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if lmList:
        cv2.circle(img, (lmList[0][0], lmList[0][1]), 6, (0, 255, 0), -1)

        lrgap, lrvel = 50, 0.5
        udgap, udvel = 35, 0.5

        if lmList[0][0] < 340:
            cv2.putText(img, '⬅ Move Left', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            vx = -1 * ((340 - lmList[0][0]) // lrgap + 1) * lrvel
        elif lmList[0][0] > 420:
            cv2.putText(img, '➡ Move Right', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            vx = ((lmList[0][0] - 420) // lrgap + 1) * lrvel

        if lmList[0][1] < 250:
            cv2.putText(img, '⬆ Move Up', (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 150), 3)
            vy = ((250 - lmList[0][1]) // udgap + 1) * udvel
        elif lmList[0][1] > 310:
            cv2.putText(img, '⬇ Move Down', (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 150), 3)
            vy = -1 * ((lmList[0][1] - 310) // udgap + 1) * udvel

    if bboxInfo:
        bbox = bboxInfo["bbox"]
        A, B, C = coff
        distanceCM = A * bbox[2]**2 + B * bbox[2] + C

        c, v = 25, 0.5

        if distanceCM < 125:
            cv2.putText(img, '⬅⬅ Move Backward', (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 255, 100), 3)
            vz = -1 * ((125 - distanceCM) // c) * v
        elif distanceCM > 175:
            cv2.putText(img, '➡➡ Move Forward', (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 255, 100), 3)
            vz = ((distanceCM - 175) // c + 1) * v

        cv2.putText(img, f'Distance: {int(distanceCM)} cm', (480, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    vel_mat = [vx, vy, vz]
    print("Velocity Vector:", vel_mat)

    cv2.imshow('Pose Detection Movement', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
