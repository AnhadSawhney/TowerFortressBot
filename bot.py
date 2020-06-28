import time
import cv2
import numpy as np
from mss import mss
import subprocess
import win32gui
import win32con    
import time
import math

class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return False
        
    def __neq__(self, other):
        return not self == other

    def distanceTo(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
    
    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

title = 'TowerFortressBot'

def launch():
    rect = {}
    deviceproc = subprocess.Popen('scrcpy\\scrcpy.exe '+
                            '--always-on-top '+
                            '--stay-awake '+
                            '--max-fps 60 '+
                            '--bit-rate 16M '+
                            '--max-size 780 '+
                            '--window-title '+title+'-Device ', shell=True)

    time.sleep(2)
    handle = win32gui.FindWindow(None, title+'-Device')
    rect['left'], rect['top'], right, bottom = win32gui.GetWindowRect(handle)    
    rect['left'] += 9
    rect['top'] += 32
    rect['width'] = right - rect['left'] - 9
    rect['height'] = bottom - rect['top'] - 9
    return rect, deviceproc, handle

rect, deviceproc, handle = launch()

print(rect)

playerparams = cv2.SimpleBlobDetector_Params()

playerparams.filterByColor = False
playerparams.filterByArea = True
playerparams.minArea = 1000
playerparams.maxArea = 2000
playerparams.filterByCircularity = False
playerparams.filterByConvexity = False
playerparams.filterByInertia = False

playerdetector = cv2.SimpleBlobDetector_create(playerparams)

enemyparams = cv2.SimpleBlobDetector_Params()

enemyparams.filterByColor = False
enemyparams.filterByArea = True
enemyparams.minArea = 150
enemyparams.maxArea = 300
enemyparams.filterByCircularity = True
enemyparams.minCircularity = 0.2
enemyparams.filterByConvexity = False
enemyparams.filterByInertia = False

enemydetector = cv2.SimpleBlobDetector_create(enemyparams)

blocksize = 14
viewradius = 8
viewdiameter = 2*viewradius
boxradius = viewradius*blocksize
boxdiameter = viewdiameter*blocksize

def findplayer(img): #returns coords of player and player tracker image
    #img = img[340:510]
    hsvimg = cv2.cvtColor(img[340:510], cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsvimg, (95, 100, 100), (140, 255, 255))
    keypoints = playerdetector.detect(thresh)
    thresh = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if keypoints == []:
        return Point(-1, -1), thresh
    else:
        #return Point(75, int(keypoints[0].pt[1])+170), playerimg
        return Point(int(keypoints[0].pt[0]), int(keypoints[0].pt[1])+345), thresh

def findwalls(img): #returns 16x16 matrix of tiles and wall-less image
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsvimg, (0, 80, 0), (20, 255, 180)) #wall
    thresh = np.clip(thresh+cv2.inRange(hsvimg, (140, 100, 0), (180, 245, 160)), 0, 255) #wall
    thresh[np.where((img == 0).all(axis=2))] = 255

    out = np.zeros((viewdiameter, viewdiameter))

    for x in range(viewdiameter):
        for y in range(viewdiameter):
            if np.sum(thresh[y*blocksize:(y+1)*blocksize, x*blocksize:(x+1)*blocksize]) >= 255*blocksize*blocksize*0.7:
                out[y, x] = 1
    
    return out, cv2.subtract(img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

def findenemies(img): #returns list of enemy coords
    #img = cv2.GaussianBlur(img,(3,3), cv2.BORDER_DEFAULT)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsvimg, (0, 70, 0), (30, 245, 160)) #wall
    thresh = np.clip(thresh+cv2.inRange(hsvimg, (140, 70, 0), (180, 245, 160)), 0, 255) #wall
    thresh[np.where((img == 0).all(axis=2))] = 255
    #thresh = np.copy(img)
    #thresh[thresh >= 128] = 255
    #[thresh < 128] = 0
    #thresh = cv2.inRange(hsvimg, (45, 0, 60), (80, 255, 255)) #green
    #thresh = np.clip(thresh+cv2.inRange(hsvimg, (0, 0, 60), (5, 255, 255)), 0, 255)
    #thresh = np.clip(thresh+cv2.inRange(hsvimg, (135, 0, 60), (180, 255, 255)), 0, 255)
    #keypoints = enemydetector.detect(thresh)
    #thresh = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img

def drawview(arr):
    out = np.zeros((boxdiameter, boxdiameter, 3))
    for x in range(viewdiameter):
        for y in range(viewdiameter):
            if arr[y, x] == 1:
                out[y*blocksize:(y+1)*blocksize, x*blocksize:(x+1)*blocksize, 2] = 255

    return out

playercoords = Point(150, 400)
firstrun = True

with mss() as sct:
    # mon = sct.monitors[0]
    while True:
        last_time = time.time()
        img = np.array(sct.grab(rect))[:, :, :3]
        img = cv2.bilateralFilter(img, 5, 75, 75) 
        tempcoords, playerimg = findplayer(np.copy(img))
        if tempcoords != Point(-1, -1):
            if abs(tempcoords.y-playercoords.y) > 5:
                playercoords.y = tempcoords.y
            if abs(tempcoords.x-playercoords.x) > 1:
                playercoords.x = tempcoords.x
            #if tempcoords.distanceTo(playercoords) < 30 or firstrun:
                #playercoords = tempcoords            
        
        if playercoords.x-boxradius < 0:
            width = playercoords.x+boxradius
            cropimg = np.zeros((boxdiameter, boxdiameter, 3), dtype=np.uint8)
            cropimg[:, boxdiameter-width:] = np.copy(img[playercoords.y-boxradius:playercoords.y+boxradius, 0:width])
        elif playercoords.x+boxradius > rect['width']:
            width = rect['width']-playercoords.x+boxradius
            cropimg = np.zeros((boxdiameter, boxdiameter, 3), dtype=np.uint8)
            cropimg[:, :width] = img[playercoords.y-boxradius:playercoords.y+boxradius, playercoords.x-boxradius:rect['width']]
        else:
            cropimg = np.copy(img[playercoords.y-boxradius:playercoords.y+boxradius, playercoords.x-boxradius:playercoords.x+boxradius])

        for x in range(viewdiameter):
            for y in range(viewdiameter):
                cropimg[y*blocksize:y*blocksize+1, x*blocksize:x*blocksize+1] = (255, 255, 255)
        
        arr, cropimg = findwalls(cropimg)

        view = drawview(arr)

        enemyimg = findenemies(cropimg)
        
        #print('fps: {0}'.format(1 / (time.time()-last_time)))
        #print(np.shape(img[playercoords.y-64:playercoords.y+64, playercoords.x-64:playercoords.x+64]))
        cv2.imshow('test', cropimg)
        cv2.imshow('player', playerimg)
        cv2.imshow('enemies', enemyimg)
        cv2.imshow('view', view)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imwrite('image.png',cropimg)
            cv2.destroyAllWindows()
            deviceproc.kill()
            win32gui.PostMessage(handle,win32con.WM_CLOSE,0,0)
            break

        firstrun = False
