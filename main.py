#ur majur legue scrub
import cv2
import numpy as np
#Global variables
lastMouseClickX = -1
lastMouseClickY = -1

hTracklist = [0]
sTracklist = [0]
vTracklist = [0]

counter = 0

line = []
record = []
isLine = False
recording = False
showing = False

#mouse callback function

def mouse_handle(event,x,y,flags,param):
    global lastMouseClickX
    global lastMouseClickY
    if event == cv2.EVENT_LBUTTONDOWN:
        lastMouseClickX = x
        lastMouseClickY = y

#full processing of the camera image into contours

def process_image(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contourImgs = []
    for i in range(len(hTracklist)):
        tempLow = np.array([hTracklist[i]-5, sTracklist[i]-50, vTracklist[i]-50], dtype=np.uint8)
        if tempLow[1] > sTracklist[i]:
            tempLow[1] = 0
        if tempLow[2] > vTracklist[i]:
            tempLow[2] = 0
        if tempLow[0] < 0:
            tempLow[0] = 180+tempLow[0]
        elif tempLow[0] > 180:
            tempLow[0] = tempLow[0]-180
        tempHigh = np.array([hTracklist[i]+5, sTracklist[i]+50, vTracklist[i]+50], dtype=np.uint8)
        if tempHigh[1] < sTracklist[i]:
            tempHigh[1] = 255
        if tempHigh[2] < vTracklist[i]:
            tempHigh[2] = 255
        if tempHigh[0] < 0:
            tempHigh[0] = 180+tempHigh[0]
        elif tempHigh[0] > 180:
            tempHigh[0] = tempHigh[0]-180
        tempImg = cv2.inRange(hsv, tempLow, tempHigh)
        if tempLow[0]>tempHigh[0]:
            u1 = np.array([180, tempHigh[1], tempHigh[2]], dtype=np.uint8)
            tempImg = cv2.add(tempImg, cv2.inRange(hsv, tempLow, u1))
            l1 = np.array([0, tempLow[1], tempLow[2]], dtype=np.uint8)
            tempImg = cv2.add(tempImg, cv2.inRange(hsv, l1, tempHigh))
        tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_OPEN, (5, 5), iterations=5)
        image, contours, hierarchy = cv2.findContours(tempImg, 1, 2)
        
        contourImgs.append(contours)
    return contourImgs, hsv

#Function that sorts the contours

def sort_contours(contourList):
    for i in range(len(contourList)):
        contourList[i] = sorted(contourList[i], key=lambda contour: cv2.contourArea(contour))
    return contourList

#Draws the boxs and centers on the image using the set of contours

def draw_contours(img, contourSet):
    for i in range(len(contourSet)):
        if contourSet[i] != []:
            temp = contourSet[i][-1]
            x,y,w,h = cv2.boundingRect(temp)
            M = cv2.moments(temp)
            cx = 0
            cy = 0
            
            if(not M['m00']==0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            img = cv2.circle(img, (cx, cy), 10, (255, 255, 0))
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
    return img

#The actual script

vidCap = cv2.VideoCapture(0)
cv2.namedWindow('Webcam')
cv2.resizeWindow('Webcam',720,1280)
cv2.setMouseCallback('Webcam', mouse_handle)

while True:
    ret, img = vidCap.read()
    x = cv2.waitKey(10)
    if not len(hTracklist)==0 or lastMouseClickX != -1 or (x>-1 and chr(x)=='q'):
        
        
        if lastMouseClickX != -1:
            (hTrackTemp, sTrackTemp, vTrackTemp) = hsv[lastMouseClickY][lastMouseClickX]
            if counter==0:
                hTracklist.pop(0)
                sTracklist.pop(0)
                vTracklist.pop(0)
                counter = counter + 1
            
            hTracklist.append(hTrackTemp)
            sTracklist.append(sTrackTemp)
            vTracklist.append(vTrackTemp)
                
            lastMouseClickX = -1
            lastMouseClickY = -1
        elif x>-1 and chr(x)=="q":
            break
        elif x>-1 and chr(x)=="m":
            hTracklist.pop(-1)
            sTracklist.pop(-1)
            vTracklist.pop(-1)
        setOfContours, hsv = process_image(img)
        sortedSetOfContours = sort_contours(setOfContours)
        img = draw_contours(img, sortedSetOfContours)
        cv2.imshow('Webcam', img)        
    else:
        cv2.imshow('Webcam', img)
vidCap.release()
cv2.destroyAllWindows()        
    
