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
    deviceproc = subprocess.Popen('scrcpy-win64-v1.14\\scrcpy.exe '+
                            '--always-on-top '+
                            '--stay-awake '+
                            '--max-fps 60 '+
                            '--bit-rate 16M '+
                            '--max-size 334 '+
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
playerparams.minArea = 100
playerparams.maxArea = 500
playerparams.filterByCircularity = False
playerparams.filterByConvexity = False
playerparams.filterByInertia = False

playerdetector = cv2.SimpleBlobDetector_create(playerparams)

#enemydetector = cv2.SimpleBlobDetector_create(self.config['blob_det_params'])

def findplayer(img): #returns coords of player and player tracker image
    playerimg = np.copy(img[170:255])
    playerimg[playerimg[:,:,0] >= 110] = [255, 255, 255]
    playerimg[playerimg[:,:,0] < 110] = [0, 0, 0]
    
##    bwimg = np.zeros(np.shape(playerimg)[0:2], dtype=np.uint8)
##    #bwimg[np.where(playerimg[:,:,2] > 50)] = 1 #some red
##    #bwimg[np.where(playerimg[:,:,2] < 50)] = 1 #some red
##    bwimg = np.where(playerimg[:,:,0] >= 110) #lots of blue
##    playerimg[bwimg]
##    playerimg = cv2.cvtColor(bwimg, cv2.COLOR_GRAY2BGR); 
    keypoints = playerdetector.detect(playerimg)
    playerimg = cv2.drawKeypoints(playerimg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if keypoints == []:
        return Point(-1, -1), playerimg
    else:
        #return Point(75, int(keypoints[0].pt[1])+170), playerimg
        return Point(int(keypoints[0].pt[0]), int(keypoints[0].pt[1])+170), playerimg

#def findenemies(img): #returns list of enemy coords

playercoords = Point(75, 200)
firstrun = True

with mss() as sct:
    # mon = sct.monitors[0]
    while True:
        last_time = time.time()
        img = np.array(sct.grab(rect))[:, :, :3]
        tempcoords, playerimg = findplayer(img)
        if tempcoords != Point(-1, -1):
            if tempcoords.distanceTo(playercoords) < 20 or firstrun:
                playercoords = tempcoords            
        
        if playercoords.x-64 < 0:
            width = playercoords.x+64
            cropimg = np.zeros((128, 128, 3), dtype=np.uint8)
            cropimg[:, 128-width:] = np.copy(img[playercoords.y-64:playercoords.y+64, 0:width])
        elif playercoords.x+64 > rect['width']:
            width = rect['width']-playercoords.x+64
            cropimg = np.zeros((128, 128, 3), dtype=np.uint8)
            cropimg[:, :width] = img[playercoords.y-64:playercoords.y+64, playercoords.x-64:rect['width']]
        else:
            cropimg = np.copy(img[playercoords.y-64:playercoords.y+64, playercoords.x-64:playercoords.x+64])

        
        
        #print('fps: {0}'.format(1 / (time.time()-last_time)))
        #print(np.shape(img[playercoords.y-64:playercoords.y+64, playercoords.x-64:playercoords.x+64]))
        cv2.imshow('test', cropimg)
        cv2.imshow('test2', playerimg)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imwrite('image.png',img)
            cv2.destroyAllWindows()
            deviceproc.kill()
            win32gui.PostMessage(handle,win32con.WM_CLOSE,0,0)
            break

        firstrun = False
