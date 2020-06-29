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
playerparams.maxArea = 2500
playerparams.filterByCircularity = False
playerparams.filterByConvexity = False
playerparams.filterByInertia = False

playerdetector = cv2.SimpleBlobDetector_create(playerparams)

enemyparams = cv2.SimpleBlobDetector_Params()

enemyparams.filterByColor = False
enemyparams.filterByArea = True
enemyparams.minArea = 150
enemyparams.maxArea = 800
enemyparams.filterByCircularity = False
enemyparams.filterByConvexity = False
enemyparams.filterByInertia = False

enemydetector = cv2.SimpleBlobDetector_create(enemyparams)

blocksize = 14
viewradius = 8
viewdiameter = 2*viewradius
boxradius = viewradius*blocksize
boxdiameter = viewdiameter*blocksize

def findplayer(img, arr): #returns coords of player
    #playerlessimg = np.copy(img)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsvimg, (95, 100, 100), (140, 255, 255))

    arr[viewradius-2:viewradius+2, viewradius-2:viewradius+2] = [[0, 1, 1, 0],
                                                                 [1, 1, 1, 1],
                                                                 [1, 1, 1, 1],
                                                                 [0, 1, 1, 0]]
    
    keypoints = playerdetector.detect(thresh)
    thresh = cv2.drawKeypoints(thresh, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if keypoints == []:
        return Point(-1, -1), thresh#, playerlessimg
    else:
        #return Point(75, int(keypoints[0].pt[1])+170), playerimg
        return Point(int(keypoints[0].pt[0]), int(keypoints[0].pt[1])), thresh#, playerlessimg

def findwalls(img, offset, arr): #returns matrix of tiles and wall-less image
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsvimg, (0, 80, 0), (20, 255, 185)) #wall
    thresh = np.clip(thresh+cv2.inRange(hsvimg, (140, 100, 0), (180, 245, 160)), 0, 255) #wall
    thresh[np.where((img == 0).all(axis=2))] = 255

    for x in range(viewdiameter):
        for y in range(viewdiameter):
            if arr[y, x] == 0: #only in empty tiles
                xmin = x*blocksize+offset.x #only need to worry about x, y is always in bounds
                xmax = xmin+blocksize
                if xmin < 0: #out of bounds on left
                    arr[y, x] = 2 #wall here            
                elif xmax > rect['width']: #out of bounds on right
                    arr[y, x] = 2 #wall here
                elif np.sum(thresh[y*blocksize+offset.y:(y+1)*blocksize+offset.y, xmin:xmax]) >= 255*blocksize*blocksize*0.7:
                    arr[y, x] = 2
        
    return arr, cv2.subtract(img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

def findenemies(img, offset, arr): #given wall-less image, returns list of enemy coords
    #img = cv2.GaussianBlur(img,(3,3), cv2.BORDER_DEFAULT)
    #thresh = np.copy(img)
    #thresh[thresh >= 128] = 255
    #thresh[thresh < 128] = 0


    #TODO: Find enemies just based on everything in the image that is bright enough, except chests and torches
    
    
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = np.zeros(np.shape(img)[0:2])
    thresh[img[:,:,2] > 140] = 255 #reds of eyes
    thresh[img[:,:,0] > 5] = 0
    thresh[img[:,:,1] > 5] = 0
    thresh = cv2.inRange(hsvimg, (45, 1, 60), (80, 255, 255)) #green
    thresh = np.clip(thresh+cv2.inRange(hsvimg, (135, 1, 60), (180, 255, 255)), 0, 255) #whites of eyes
    thresh = np.clip(thresh+cv2.inRange(hsvimg, (75, 100, 100), (110, 145, 145)), 0, 255) #spider legs
    keypoints = enemydetector.detect(thresh)
    thresh = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img#thresh

def drawview(arr):
    out = np.zeros((boxdiameter, boxdiameter, 3))
    for x in range(viewdiameter):
        for y in range(viewdiameter):
            if arr[y, x] == 1: #player
                out[y*blocksize:(y+1)*blocksize, x*blocksize:(x+1)*blocksize, 0] = 255
            if arr[y, x] == 2: #walls
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
        img = img[100:610] #crop out ui
        arr = np.zeros((viewdiameter, viewdiameter))
        tempcoords, playerimg = findplayer(img, arr) 
        if tempcoords != Point(-1, -1):
            if abs(tempcoords.y-playercoords.y) > 4:
                playercoords.y = tempcoords.y
            if abs(tempcoords.x-playercoords.x) > 1:
                playercoords.x = tempcoords.x
            #if tempcoords.distanceTo(playercoords) < 30 or firstrun:
                #playercoords = tempcoords

        offset = Point(round(playercoords.x/blocksize)*blocksize - boxradius, round(playercoords.y/blocksize)*blocksize - boxradius) #round to nearest boxsize, subtract viewradius
       
        arr, walllessimg = findwalls(img, offset, arr)
        playerlessimg = cv2.subtract(walllessimg, playerimg)
        enemyimg = findenemies(playerlessimg, offset, arr)
        view = drawview(arr)

        maxy, maxx = np.shape(walllessimg)[0:2]
        for x in range(viewdiameter):
            for y in range(viewdiameter):
                draw = Point(x*blocksize+offset.x, y*blocksize+offset.y)
                if draw.y < maxy and draw.x > 0 and draw.x < maxx:
                    walllessimg[draw.y, draw.x] = (255, 255, 255)
        
##        if playercoords.x-boxradius < 0:
##            width = playercoords.x+boxradius
##            cropimg = np.zeros((boxdiameter, boxdiameter, 3), dtype=np.uint8)
##            cropimg[:, boxdiameter-width:] = np.copy(img[playercoords.y-boxradius:playercoords.y+boxradius, 0:width])
##        elif playercoords.x+boxradius > rect['width']:
##            width = rect['width']-playercoords.x+boxradius
##            cropimg = np.zeros((boxdiameter, boxdiameter, 3), dtype=np.uint8)
##            cropimg[:, :width] = img[playercoords.y-boxradius:playercoords.y+boxradius, playercoords.x-boxradius:rect['width']]
##        else:
##            cropimg = np.copy(img[playercoords.y-boxradius:playercoords.y+boxradius, playercoords.x-boxradius:playercoords.x+boxradius])
##
##        for x in range(viewdiameter):
##            for y in range(viewdiameter):
##                cropimg[y*blocksize:y*blocksize+1, x*blocksize:x*blocksize+1] = (255, 255, 255)
        
        #print('fps: {0}'.format(1 / (time.time()-last_time)))
        #print(np.shape(img[playercoords.y-64:playercoords.y+64, playercoords.x-64:playercoords.x+64]))
        #cv2.imshow('crop', cropimg)
        cv2.imshow('walls', walllessimg)
        cv2.imshow('player', playerimg)
        cv2.imshow('enemies', enemyimg)
        cv2.imshow('view', view)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            #cv2.imwrite('image.png',cropimg)
            cv2.destroyAllWindows()
            deviceproc.kill()
            win32gui.PostMessage(handle,win32con.WM_CLOSE,0,0)
            break

        firstrun = False
