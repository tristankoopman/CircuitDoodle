import numpy
import scipy
import cv2
import csv
from sklearn.decomposition import PCA
from sklearn.svm import SVC

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

#convert image to csv line of pixel darkness and predict circuit component

def storeAlts(imageA,firstval):
    o = open("trainData.csv", "a")
    data = imageA
    points = [-4,0,4]
    for x in points:
        temp = data[-x:]+data[:-x]
        newData = []
        newData.append(firstval)
        for pix in temp:
            newData.append(pix)
        newData.append(sum(newData)-firstval)
        o.write(",".join(map(str,newData))+"\n")
    o.close

def get30by30(xpix,ypix,imag):
    imageArray = []
    currentdarkness = 0;
    for x in range(0, 30):
        for y in range(0, 30):
            cornerX = xpix + x*8
            cornerY = ypix + y*8
            currentdarkness = 0;
            for j in range(0,8):
                for k in range(0,8):
                    R,G,B= imag[cornerX + j, cornerY + k]
                    currentdarkness += 255 - sum([R,G,B])/3
            currentdarkness = currentdarkness//64
            if currentdarkness>30:
                imageArray.append(currentdarkness)
            else:
                imageArray.append(0)
    return imageArray
#convert image to csv line of pixel darkness and predict circuit component
def save(imgnew):
    #get new images data
    imageCSV = []
    img0 = cv2.imread(imgnew)
    img1 = numpy.rot90(img0)
    img2 = numpy.rot90(img1)
    img3 = numpy.rot90(img2)
    image0 = get30by30(0,0,img0)
    image1 = get30by30(0,0,img1)
    image2 = get30by30(0,0,img2)
    image3 = get30by30(0,0,img3)
    #imageCSV.append(sum(imageCSV))
    #check  for label
    print "What is the label for this component? press 'v' or 'i' or 'w' or 'r' or 'e' or 'g'"
    correctLabel=cv2.waitKey(0)&0xFF
    if correctLabel==103:
        newData = [ord('G')]
    if correctLabel==101:
        newData = [ord('E')]
    if correctLabel==118:
        newData = [ord('V')]
    if correctLabel==105:
        newData = [ord('I')]
    if correctLabel==119:
        newData = [ord('W')]
    if correctLabel==114:
        newData = [ord('R')]
    
    storeAlts(image0,newData[0])
    storeAlts(image1,newData[0])
    storeAlts(image2,newData[0])
    storeAlts(image3,newData[0])
    print "Thank you, Draw a new component!"
    
# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
    	drawing=True
    	ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE:
    	if drawing==True:
    	   if mode==True:
            cv2.line(img,(ix,iy),(x,y),(0,0,0),5)
            ix=x
            iy=y
    elif event==cv2.EVENT_LBUTTONUP:
    	drawing=False
	if mode==True:
            cv2.line(img,(ix,iy),(x,y),(0,0,0),5)
            ix=x
            iy=y

#loop
img = cv2.imread("whitePage.png",0)
cv2.namedWindow('circuitComponent')
cv2.setMouseCallback('circuitComponent',interactive_drawing)
while(1):
    cv2.imshow('circuitComponent',img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    if k==112:
        cv2.imwrite( "resized.jpg",img);
        save("resized.jpg")
        img = cv2.imread("whitePage.png",0)
        
cv2.destroyAllWindows()
