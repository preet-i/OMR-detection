import cv2
import numpy as np
import utils

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    print(add)
    print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew
def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx
def splitBoxes(img):
    rows = np.vsplit(img,5)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes
def showAnswers(img,myIndex,grading,ans,questions=5,choices=5):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for x in range(0,questions):
        myAns= myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        
        if grading[x]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)
            correctAns=ans[x]
            cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+secH//2), 20,(0,255,0),cv2.FILLED)

            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
        cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
    return img    
  
#################################################################################################

path="1.jpg"
widthImg=700
heightImg=700
questions=5
choices=5
ans=[1,2,0,1,4]
webcamFeed=False
cameraNo=0
#########################################################
cap=cv2.VideoCapture(cameraNo)
cap.set(10,150)
while True:
    if webcamFeed:success,img=cap.read()
    else:img=cv2.imread(path)

    img= cv2.imread(path)
    img=cv2.resize(img,(heightImg,widthImg))
    imgContours=img.copy()
    imgFinal=img.copy()
    imgBiggestContours=img.copy()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgBlur,10,50)

    try:
        contours, hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

        rectCon=rectContour(contours)
        biggestContour=getCornerPoints(rectCon[0])
        gradePoints=getCornerPoints(rectCon[1])

        if biggestContour.size!=0 and gradePoints.size!=0:
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
            cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)
            biggestContour=reorder(biggestContour)
            gradePoints=reorder(gradePoints)
            
            pt1=np.float32(biggestContour)
            pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix=cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
            
            ptG1=np.float32(gradePoints)
            ptG2=np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG=cv2.getPerspectiveTransform(ptG1,ptG2)
            imgGradeDisplay=cv2.warpPerspective(img,matrixG,(325,150))
            #cv2.imshow("Grade",imgGradeDisplay)

        imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgThresh=cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

        boxes=splitBoxes(imgThresh)
        # cv2.imshow("Test",boxes[2])
        # print(cv2.countNonZero(boxes[2]))
        myPixelVal=np.zeros((questions,choices))
        countC=0
        countR=0

        for image in boxes:
            totalPixels=cv2.countNonZero(image)
            myPixelVal[countR][countC]=totalPixels
            countC+=1
            if(countC==choices):countR+=1 ;countC=0
        print(myPixelVal)

        myIndex=[]
        for x in range(0,questions):
            arr=myPixelVal[x]
            # print("arr",arr)
            myIndexVal=np.where(arr==np.amax(arr))
            #print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
        print(myIndex)


        grading=[]
        for x in range(0,questions):
            if ans[x]==myIndex[x]:
                grading.append(1)
            else:grading.append(0)
            # print(grading)
        score=sum(grading)/questions*100
        print(score)
        imgResult=imgWarpColored.copy()
        imgResult=showAnswers(imgResult,myIndex,grading,ans,questions,choices)
        imgRawDrawing=np.zeros_like(imgWarpColored)
        imgRawDrawing=showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
        invMatrix=cv2.getPerspectiveTransform(pt2,pt1)
        imgInvWarp=cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))

        imgBlank=np.zeros_like(img)
        imgRawGrade=np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade,str(int(score))+'%',(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
        invmatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
        imgInvGradeDisplay=cv2.warpPerspective(imgRawGrade,invmatrixG,(widthImg,heightImg))

        imgFinal=cv2 .addWeighted(imgFinal,1,imgInvWarp,1,0)
        imgFinal=cv2 .addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)
        cv2.imshow("x",imgFinal)

        
        imageArray=([img,imgGray,imgBlur,imgCanny],[imgContours,imgBiggestContours,imgWarpColored,imgThresh],[imgResult,imgRawDrawing,imgInvWarp,imgFinal])
    except:
        imageArray=([img,imgGray,imgBlur,imgCanny],[imgBlank,imgBlank,imgBlank,imgBlank],[imgBlank,imgBlank,imgBlank,imgBlank])

    labels=[["Original","Gray","Blur","Canny"],["Contour","Biggest Cont","Warp","Thresh"],["Result","RawDrawing","Inverse Warp","Final Image"],]
    imgStacked=stackImages(imageArray,0.3,labels)

    cv2.imshow("Stacked",imgStacked)
    if cv2.waitKey(0) & 0xFF==ord('s'):
        cv2.imwrite("FinalResult.jpg",imgFinal)
        cv2.waitKey(300)