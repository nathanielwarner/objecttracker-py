import cv2
import numpy as np
import math
import random
import time

#Global variables
lastMouseClickX = -1
lastMouseClickY = -1

hTracklist = [0]
sTracklist = [0]
vTracklist = [0]

counter = 0
selected = 0

distWeight = 0.1

isBeginning = True

line = []
record = []
isLine = False
lineNum = 0
recordNum = 0
recording = False
showing = False
allMode = False
isLive = True

recordList = []
colorList = []

cxList = [0]
cyList = [0]

distXList = [0]
distYList = [0]

deltaDistXList = [0]
deltaDistYList = [0]

highestScores = [0]

hueRecal = False

lastTime = 0

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
        tempLow = np.array([hTracklist[i]-7, sTracklist[i]-70, vTracklist[i]-70], dtype=np.uint8)
        if tempLow[1] > sTracklist[i]:
            tempLow[1] = 0
        if tempLow[2] > vTracklist[i]:
            tempLow[2] = 0
        if tempLow[0] < 0:
            tempLow[0] = 180+tempLow[0]
        elif tempLow[0] > 180:
            tempLow[0] = tempLow[0]-180
        tempHigh = np.array([hTracklist[i]+7, sTracklist[i]+70, vTracklist[i]+70], dtype=np.uint8)
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
    global line, cxList, cyList, doNotOverwriteLastCoords, distWeight
    
    #Loops through the first dimension of the contourSet (a set of lists of contours)
    for i in range(len(contourSet)):
        if contourSet[i] != []:
            lastCx = cxList[i]
            lastCy = cyList[i] 
            
            expectedCx = lastCx + distXList[i] + int(deltaDistXList[i]*1)
            
            if expectedCx < 0:
                expectedCx = 0
            elif expectedCx > img.shape[1]:
                expectedCx = img.shape[1]
                
            expectedCy = lastCy + distYList[i] + int(deltaDistYList[i]*1)
            if expectedCy < 0:
                expectedCy = 0
            elif expectedCy > img.shape[0]:
                expectedCy = img.shape[0]
               
            highestScore = 0
            bestContour = None
            bestCx = lastCx
            bestCy = lastCy
            for j in range(len(contourSet[i])):
                temp = contourSet[i][j]
                area = cv2.contourArea(temp)
                
                M = cv2.moments(temp)
                if(not (M['m00']==0)):
                    tempCx = int(M['m10']/M['m00'])
                    tempCy = int(M['m01']/M['m00'])
                else:
                    tempCx = 0
                    tempCy = 0
                dist = math.sqrt((expectedCx - tempCx)**2 + (expectedCy - tempCy)**2)
                distPow = dist**2
                score = (1-distWeight) * area - distWeight * distPow
                if score > highestScore:
                    highestScore = score
                    bestContour = temp
                    bestCx = tempCx
                    bestCy = tempCy
            #print str(highestScore)
            highestScores[i] = highestScore
            prevDistX = distXList[i]
            prevDistY = distYList[i]
            
            distXList[i] = bestCx - lastCx
            distYList[i] = bestCy - lastCy
            
            deltaDistXList[i] = distXList[i] - prevDistX
            deltaDistYList[i] = distYList[i] - prevDistY
                    
            x,y,w,h = cv2.boundingRect(bestContour)
            M = cv2.moments(bestContour)
            if(not (M['m00']==0)):
                cxList[i] = int(M['m10']/M['m00'])
                cyList[i] = int(M['m01']/M['m00'])              
            
            if not (i==selected):
                img = cv2.circle(img, (cxList[i], cyList[i]), 10, (255, 255, 0))
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)   
                #Prediction
                #img = cv2.circle(img, (expectedCx, expectedCy), 10, (255,0,255))
                #Last position
                #img = cv2.circle(img, (lastCx, lastCy), 10, (255,255,0))                
            else:
                img = cv2.circle(img, (cxList[i], cyList[i]), 10, (0, 0, 255))
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
                #Prediction
                #img = cv2.circle(img, (expectedCx, expectedCy), 10, (0,255,255)) 
                #Last position
                #img = cv2.circle(img, (lastCx, lastCy), 10, (0,0,255))                  
            if isLine and i==lineNum:
                line.append([cxList[i], cyList[i]])
                if len(line)>25:
                    line.pop(0)
            elif len(line)>0 and i==lineNum:
                line.pop(0)
            if len(line)>1:
                for b in range(len(line)-1):
                    img = cv2.line(img, (line[b][0], line[b][1]), (line[b+1][0], line[b+1][1]), (0, 150, 150))
            if not allMode:
                if recording and i==recordNum:
                    record.append([cxList[i], cyList[i]])
                if len(record)>1 and showing:
                    for i in range(len(record)-1):
                        img = cv2.line(img, (record[i][0], record[i][1]), (record[i+1][0], record[i+1][1]), (150, 0, 150))                
            else:
                if recording:
                    recordList[i].append([cxList[i], cyList[i]])
                if len(recordList)>1 and len(recordList[i])>1 and showing:
                    
                    for j in range(len(recordList[i])-1):
                        img = cv2.line(img, (recordList[i][j][0], recordList[i][j][1]), (recordList[i][j+1][0], recordList[i][j+1][1]), (colorList[i][0], colorList[i][1], colorList[i][2])) 
    return img


def displayText():
    global img
    
    start = 30
    offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0,0,255)
    
    cv2.putText(img, "Click an object to start tracking it", (start,int(img.shape[0]-start)), font, scale, color)
    cv2.putText(img, "Press q to quit", (start,start), font, scale, color)      
    if isLine:
        cv2.putText(img, "Press t to turn off line tracking", (start,start + 1*offset), font, scale, color)
    else:
        cv2.putText(img, "Press t to turn on line tracking", (start,start + 1*offset), font, scale, color)
    if recording:
        cv2.putText(img, "Press r to stop recording", (start,start + 2*offset), font, scale, color)
    else:
        cv2.putText(img, "Press r to start recording", (start, start + 2*offset), font, scale, color)
    if showing:
        cv2.putText(img, "Press p to hide the recording", (start, start + 3*offset), font, scale, color)
    else:
        cv2.putText(img, "Press p to show the recording", (start, start + 3*offset), font, scale, color)
        
    cv2.putText(img, "Press c to clear lines and record", (start,start + 4*offset), font, scale, color)
        
    if hueRecal:
        cv2.putText(img, "Press b to disable real-time hsv recalibration for all objects", (start,start + 5*offset), font, scale, color)
    else:
        cv2.putText(img, "Press b to enable real-time hsv recalibration for all objects", (start, start + 5*offset), font, scale, color)
    cv2.putText(img, "Press h to recalibrate hsv for the selected object now", (start,start + 6*offset), font, scale, color)
    cv2.putText(img, "Press m to deselect the current object", (start, start + 7*offset), font, scale, color)
    cv2.putText(img, "Press i to select the next object", (start, start + 8*offset), font, scale, color)
    cv2.putText(img, "Press k to select the previous object", (start, start + 9*offset), font, scale, color)
    cv2.putText(img, "Current distWeight is: " + str(distWeight), (start, start + 10*offset), font, scale, color)
    if allMode:
        cv2.putText(img, "Your action will effect ALL tracked objects", (start, start + 11*offset), font, scale, color)
    else:
        cv2.putText(img, "Your action will effect ONLY THE SELECTED object", (start, start + 11*offset), font, scale, color)

#The actual script

#vidCap = cv2.VideoCapture("juggling.mp4")
vidCap = cv2.VideoCapture(0)


cv2.namedWindow('Webcam')
cv2.resizeWindow('Webcam',720,1280)
cv2.setMouseCallback('Webcam', mouse_handle)

while True:
    #vidCap.set(5, 60)
    ret, img = vidCap.read()
    x = cv2.waitKey(10)
    if not (len(hTracklist)==0) or lastMouseClickX != -1 or (x>-1 and chr(x)=="q"):
        setOfContours, hsv = process_image(img)
        sortedSetOfContours = sort_contours(setOfContours)
        img = draw_contours(img, sortedSetOfContours)      
        if hueRecal:
            for i in range(len(hTracklist)):
                if not highestScores[i] == 0:
                    (hNew,sNew,vNew) = hsv[cyList[i]][cxList[i]]
                    hTracklist[i] = hNew
                    sTracklist[i] = sNew
                    vTracklist[i] = vNew

        if lastMouseClickX != -1:
            (hTrackTemp, sTrackTemp, vTrackTemp) = hsv[lastMouseClickY][lastMouseClickX]
            if counter==0 and isBeginning:
                hTracklist.pop(0)
                sTracklist.pop(0)
                vTracklist.pop(0)
                cxList.pop(0)
                cyList.pop(0)
                distXList.pop(0)
                distYList.pop(0)
                deltaDistXList.pop(0)
                deltaDistYList.pop(0)
                highestScores.pop(0)
                
                counter = counter + 1
                isBeginning = False
            
            cxList.append(lastMouseClickX) 
            cyList.append(lastMouseClickY) 
            hTracklist.append(hTrackTemp)
            sTracklist.append(sTrackTemp)
            vTracklist.append(vTrackTemp)
            recordList.append([])
            distXList.append(0)
            distYList.append(0)
            deltaDistXList.append(0)
            deltaDistYList.append(0)
            highestScores.append(0)
            tempR = random.randrange(0, 255)
            tempB = random.randrange(0, 255)
            tempG = random.randrange(0, 255)
            colorList.append([tempB, tempG, tempR])
                
            lastMouseClickX = -1
            lastMouseClickY = -1
        elif x>-1 and chr(x)=="q":
            break
        elif x>-1 and chr(x)=="m" and len(hTracklist)>0 and selected>=0 and selected <len(hTracklist):
            hTracklist.pop(selected)
            sTracklist.pop(selected)
            vTracklist.pop(selected)
            cxList.pop(selected)
            cyList.pop(selected)
            if not isBeginning: 
                recordList.pop(selected)
                colorList.pop(selected)
            distXList.pop(selected)
            distYList.pop(selected)
            deltaDistXList.pop(selected)
            deltaDistYList.pop(selected)
            highestScores.pop(selected)
            counter -= 1
        elif x>-1 and chr(x)=="i" and selected<len(hTracklist)-1 and len(hTracklist)>0:
            selected = selected + 1
        elif x>-1 and chr(x)=="k" and selected>0 and len(hTracklist)>0:
            selected = selected - 1
        elif x>-1 and chr(x)=="t":
            if isLine:
                isLine = False
            else:
                isLine = True
                lineNum = selected
        elif x>-1 and chr(x)=="r":
            if recording:
                recording = False
            else:
                recording = True
                recordNum = selected
        elif x>-1 and chr(x)=="p":
            if showing:
                showing = False
            else:
                showing = True
        elif x>-1 and chr(x)=="h" and not hueRecal:
            (hNew,sNew,vNew) = hsv[cyList[selected]][cxList[selected]]
            hTracklist[selected] = hNew
            sTracklist[selected] = sNew
            vTracklist[selected] = vNew
        elif x>-1 and chr(x)=="c":
            line = []
            record = []
            for i in range(len(recordList)):
                recordList[i] = []
        elif x>-1 and chr(x)=="b":
            if hueRecal:
                hueRecal = False
            else:
                hueRecal = True
        elif x>-1 and chr(x)=="a":
            if allMode:
                allMode = False
            else:
                allMode = True
        elif x>-1 and chr(x)=="y":
            if isLive:
                isLive = False
                filename = raw_input("Enter the file name:")
                vidCap = cv2.VideoCapture(filename)
            else:
                isLive = True
                vidCap = cv2.VideoCapture(0)
                
            
    else:
        selected = -1
    if x>-1 and chr(x)=="x" and distWeight<0.98:
        distWeight += 0.02
    elif x>-1 and chr(x)=="z" and distWeight>0.02:
        distWeight -= 0.02


    displayText()
    cv2.imshow('Webcam', img)
    #print time.time() * 1000.0 - lastTime
    #lastTime = time.time() * 1000.0
vidCap.release()
cv2.destroyAllWindows()
