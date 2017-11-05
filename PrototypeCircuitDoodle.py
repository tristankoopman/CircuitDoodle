import numpy
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC


COMPONENT_NUM = 35
drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

def get30by30(xpix,ypix,image):
    imag = cv2.imread(image)
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
            imageArray.append(currentdarkness//64)
    return imageArray

#convert image to csv line of pixel darkness and predict circuit component
def predict(imgf):
    componentArray = []
    #read training data
    with open('trainData.csv', 'r') as reader:
        reader.readline()
        train_label = []
        train_data = []
        for line in reader.readlines():
            data = list(map(int, line.rstrip().split(',')))
            train_label.append(data[0])
            train_data.append(data[1:])
    #principal component analysis reducion
    train_label = numpy.array(train_label)
    train_data = numpy.array(train_data)
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    #train svm
    svc = SVC()
    svc.fit(train_data, train_label)

    #predict each square
    for row in range(0,3):
        for col in range(0,5):
            imageCSV = get30by30(row*240,col*240, imgf)
            imageCSV.append(sum(imageCSV)//900)
            test_data = numpy.array(imageCSV)
            test_data = test_data.reshape(1,-1)
            test_data = pca.transform(test_data)
            prediction = svc.predict(test_data)
            componentArray.append(prediction[0])
    return componentArray

# mouse drawing function
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
img = cv2.imread("grid.png",0)
cv2.namedWindow('circuitComponent')
cv2.setMouseCallback('circuitComponent',interactive_drawing)
while(1):
    cv2.imshow('circuitComponent',img)
    k=cv2.waitKey(1)&0xFF
    if k==27: #esc key
        break
    if k==112: #p key
        cv2.imwrite( "resized.jpg",img);
        predictionArray = predict("resized.jpg")
        break
    if k==114: #r key         restart
        img = cv2.imread("grid.png",0)       
cv2.destroyAllWindows()


#testArray = [87,82,87,87,87,86,69,69,69,82,87,87,71,87,87]
testArray = predictionArray

w,h = 5,3
checkedArrayVals = [[-1 for x in range(w)] for y in range(h)]
labelArray = [[0 for x in range(w)] for y in range(h)]
curIndex = (0,0)
curNode = 0
netList = [[0 for x in range(3)] for y in range(5)]
numOfComps = 0
#init the 2d label array
count = 0
for x in range(h):
    for y in range(w):
        labelArray[x][y] = testArray[count]
        count += 1
print "labels predicted:"
print labelArray

def findGround():
    grd = (-1,-1)
    found = 0 #false
    for x in range(h):
        for y in range(w):
            if labelArray[x][y] == 71:
                if found == 0:
                    grd = (x, y)
                    found = 1 #true
                else:
                    print "More than one ground found."
    if grd == (-1,-1):
        print "No ground found."
    checkedArrayVals[grd[0]][grd[1]] = 0
    return grd

def neighbors(index):
    x = index[0]
    y = index[1]
    listOfN = []
    if y != 0:
        listOfN.append((x,y-1))
    if y != 4:
        listOfN.append((x,y+1))
    if x != 0:
        listOfN.append((x-1,y))
    if x != 2:
        listOfN.append((x+1,y))
    return listOfN

def chooseNext(list):
    for index in list:
        label = labelArray[index[0]][index[1]] 
        if label != 69:
            if checkedArrayVals[index[0]][index[1]] == -1:
                return index
    return (-1,-1)

def chooseNextForExtend(list):
    for index in list:
        label = labelArray[index[0]][index[1]] 
        if label != 69:
            if checkedArrayVals[index[0]][index[1]] == -1:
                if label == 87:
                    return index
    return (-1,-1)

def isFinished(): #returns 0 for false, 1 for true
    for x in range(h):
        for y in range(w):
            if checkedArrayVals[x][y] == -1:
                if labelArray[x][y] != 69:
                    return 0
    return 1

def isWire(index): #returns 0 for false, 1 for true
    x = index[0]
    y = index[1]
    if labelArray[x][y] == 87:
        return 1
    else:
        return 0

def extendNode(thisNode, thisIndex):
    theNeighbors = neighbors(thisIndex)
    Index = chooseNextForExtend(theNeighbors)
    if chooseNextForExtend(theNeighbors) == (-1,-1):
        Index = thisIndex
    while(chooseNextForExtend(theNeighbors) != (-1,-1)):
        direction = chooseNextForExtend(theNeighbors)
        while(isWire(direction) == 1):
            Index = direction
            checkedArrayVals[direction[0]][direction[1]] = thisNode
            newN = neighbors(direction)
            direction = chooseNext(newN)
    return Index

def addToList(thisNode, thisIndex):
    curNode = thisNode
    checkedArrayVals[thisIndex[0]][thisIndex[1]] = labelArray[thisIndex[0]][thisIndex[1]]
    nextNodeIndex = chooseNext(neighbors(thisIndex))
    if nextNodeIndex == (-1,-1):
        netList[numOfComps][0] = labelArray[thisIndex[0]][thisIndex[1]]
        netList[numOfComps][1] = curNode
        netList[numOfComps][2] = 0 #?????
        curNode = 0#???????
        curIndex = nextNodeIndex
        return curNode,curIndex
    if checkedArrayVals[nextNodeIndex[0]][nextNodeIndex[1]] == -1:
        netList[numOfComps][0] = labelArray[thisIndex[0]][thisIndex[1]]
        netList[numOfComps][1] = curNode
        netList[numOfComps][2] = curNode + 1
        curNode+=1
        checkedArrayVals[nextNodeIndex[0]][nextNodeIndex[1]] = curNode
        curIndex = nextNodeIndex
        return curNode,curIndex

def findlastComp():
    for x in range(h):
        for y in range(w):
            if checkedArrayVals[x][y] == -1:
                if labelArray[x][y] != 69:
                    if labelArray[x][y] != 87:
                        checkedArrayVals[x][y] = labelArray[x][y]
                        return (x,y)

def lastCompNodes(listOfN):
    theNode = 0
    for index in listOfN:
        num = checkedArrayVals[index[0]][index[1]]
        if num > -1:
            if theNode == 0:
                node1 = num
                theNode += 1
            else:
                node2 = num
                return node1,node2

curIndex = findGround()
counter = 1
while(isFinished() == 0):  
    if curIndex == (-1,-1):
        curIndex = findlastComp()
        compN = neighbors(curIndex)
        node1,node2 = lastCompNodes(compN)
        netList[numOfComps][0] = labelArray[curIndex[0]][curIndex[1]]
        netList[numOfComps][1] = node1
        netList[numOfComps][2] = node2
        numOfComps +=1
    else:
        curIndex = extendNode(curNode, curIndex)
        nextComponentIndex = chooseNext(neighbors(curIndex))
        curNode,curIndex = addToList(curNode, nextComponentIndex)
        numOfComps +=1

print "Nodes found:"
print checkedArrayVals
print "Net List created:"
print netList
