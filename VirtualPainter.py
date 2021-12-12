import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

############################
brushThickness = 20
eraserThickness = 75
changeThickness = 15
#############################

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[4]
drawColor = (0, 0, 0)

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

detector = htm.handDetector(detectionCon=0.8)
xp, yp = 0,0
imgCanvas = np.zeros((1080, 1920, 3), np.uint8)

pTime = 0

while True:

    # 1 - import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2 - find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    cTime = time.time()
    deltaTime = cTime - pTime

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3 - check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4 - if selectionMode = two fingers are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)
            # print("Selection Mode")
            # checking for the selection choice
            if y1 < 200:
                # increase thickness
                if 0 < x1 < 150:
                    if changeThickness < 50:
                        if deltaTime > .5:
                            changeThickness = changeThickness + 5
                            pTime = time.time()
                # decrease thickness
                elif 150 < x1 < 300:
                    if changeThickness > 5:
                        if deltaTime > .5:
                            changeThickness = changeThickness - 5
                            pTime = time.time()
                # blue
                elif 300 < x1 < 600:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)
                # red
                elif 600 < x1 < 875:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                # green
                elif 875 < x1 < 1175:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                # yellow
                elif 1175 < x1 < 1500:
                    header = overlayList[3]
                    drawColor = (0, 255, 255)
                #eraser
                elif 1500 < x1 < 1775:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                # reset
                elif 1775 < x1 < 1919:
                    cv2.rectangle(imgCanvas, (0,0), (1919, 1079), (0,0,0), cv2.FILLED)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5 - if drawingMode = index is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, changeThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, changeThickness)

            xp, yp = x1, y1
    displayStr = "Brush Thickness: " + str(changeThickness)
    cv2.putText(img, displayStr, (10, 255), cv2.QT_FONT_NORMAL, 2, (200, 0, 200), 2)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:200,0:1920] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)