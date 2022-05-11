
import random
from turtle import width
import numpy as np
import copy
import cv2
import os
import copy
import time


#TRANS_X_STD = 0.5
TRANS_X_STD = 1.0
TRANS_Y_STD = 1.0
#TRANS_S_STD = 0.001
TRANS_Z_STD = 1.0
SHOW_ALL = 0
SHOW_SELECTED = 1

MATRIX_B1 = np.asarray([[ 849.77075856,  -69.56167559, 2146.16636095, 1046.71992891],
 [467.02973169,  -42.48548829, 1410.70732728,  400.2542614],
 [0.8110194,    -0.06639029,    2.04698316,    1]]
) 

MATRIX_B3 = np.asarray([[ 888.00928246,  -82.68559324, 2895.21561849,  606.54730596],
 [1045.53046359, -122.66897375, 4428.52498105,  -49.72861038],
 [   1.67979024,   -0.1612126,     5.67332336,    1]])

WIDTH = 100
HEIGHT = 100

A1 = 2.0
A2 = -1.0
B0 = 1.0000


class Particle(object):
    def __init__(self, x=0, y=0, z=0, xp=0, yp=0,zp=0,width=100,height=100, w=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.xp = xp
        self.yp = yp
        self.zp = zp
        self.width = width
        self.height = height
        self.w = w


class ParticleFilter(object):
    def __init__(self) -> None:
        #self.particles = []
        self.numParticles = 100
        self.n = 0

    '''粒子初始化'''
    def initiate(self, addr):
        particles = []
        x_ = addr[0]
        y_ = addr[1]
        z_ = addr[2]
        particle = Particle(x_,y_,z_,x_, y_, z_)
        for i in range(self.numParticles):
            particles.append(particle)
        return particles

    def calTransition(self, p, w, h):
        #pn = Particle()
        '''
        x = A1*(p.x-p.x0) + A2*(p.xp-p.x0)+B0 * \
            np.random.normal(0, TRANS_X_STD) + p.x0
        y = A1*(p.y-p.y0) + A2*(p.yp-p.y0)+B0 * \
            np.random.normal(0, TRANS_Y_STD) + p.y0
        s = p.s + np.random.normal(0, TRANS_S_STD)
        '''
        x = p.x-p.xp + p.x + np.random.normal(0, TRANS_X_STD)
        y = p.y-p.yp + p.y + np.random.normal(0, TRANS_Y_STD)
        z = p.z-p.zp + p.z + np.random.normal(0, TRANS_Z_STD)
        p.xp = p.x
        p.yp = p.y
        p.zp = p.z
        p.x = x
        p.y = y
        p.z = z
        p.w = 0
        return p

    '''粒子传播'''

    def transition(self, particles, w, h):
        for i in range(self.numParticles):
            particles[i] = self.calTransition(particles[i], w, h)
        return particles
    

    '''获得粒子预测区域颜色直方图'''
    def colorHist(self, imgHSV, pl, h, w):

        '''
         opencv hsv 范围:
        h(0,180)
        s(0,255)
        v(0,255)
        '''
        loc = np.asarray([pl.x,pl.y,pl.z]).astype(np.uint8)
        twoLoc_1 = np.dot(MATRIX_B1,loc)
        twoLoc_2 = np.dot(MATRIX_B3,loc)

        twoLoc_1 = twoLoc_1/twoLoc_1[2]
        twoLoc_2 = twoLoc_2/twoLoc_2[2]

        
        t = max(0, int(pl.x-pl.width/2))
        l = max(0, int(pl.y-pl.height/2))
        b = min(w, int(pl.x+pl.width/2))
        r = min(h, int(pl.y+pl.height/2))
        hsvImg = imgHSV[l:r, t:b, :]
        height,width,_ = hsvImg.shape
        H = np.zeros((height,width),dtype=np.uint8)
        S = np.zeros((height,width),dtype=np.uint8)
        V = np.zeros((height,width),dtype=np.uint8)

        h = hsvImg[..., 0]
        s = hsvImg[..., 1]
        v = hsvImg[..., 2]

        h = 2*h
        H[(h > 315) | (h <= 20)] = 0
        H[(h > 20) & (h <= 40)] = 1
        H[(h >40) & (h <= 75)] = 2
        H[(h >75) & (h <= 155)] = 3
        H[(h > 155) & (h <= 190)] = 4
        H[(h > 190) & (h <= 270)] = 5
        H[(h > 270) & (h <= 295)] = 6
        H[(h > 295) & (h <= 315)] = 7
       
        '''
        255*0.2 = 51
        255*0.7 = 178
        '''
        S[s<=51] = 0
        S[(s>51) & (s <= 178)] = 1
        S[s > 178] = 2

        V[v<=51] = 0
        V[(v>51) & (v <= 178)] = 1
        V[v > 178] = 2

        g = 9*H + 3*S + V
        hist = cv2.calcHist([g], [0], None, [72], [0, 71])
        return hist

    '''计算相似度'''
    def likelihood(self, img, pl, objecthist,h,w):
        hist = self.colorHist(img, pl,h,w)
        score = cv2.compareHist(hist,objecthist,cv2.HISTCMP_CORREL)
        return score

    '''更新粒子权重，并归一化'''
    def updateweight(self, particles, image, objecthist,h,w):
        total= 0.0
        for p in particles:
            p.w = self.likelihood(image, p, objecthist,h,w)
            total += p.w
        for p in particles:
            p.w /= total
        #particles = sorted(particles,key = lambda m:m.w,reverse=True)
        return particles

    '''重采样'''
    def resample(self, particles):
        n = self.numParticles
        k = 0
        particles = sorted(particles,key = lambda m:m.w,reverse=True)
        new_particles = copy.deepcopy(particles)
        for i in range(n):
            np = round(particles[i].w*n)
            for j in range(np):
                new_particles[k] = particles[i]
                k = k+1
                if(k==n):
                    break
            if(k==n):
                break
        while k<n:
            new_particles[k] = particles[0]
            k = k+1
        #particles = copy.deepcopy(new_particles)
        return new_particles
        '''
        new_particles = []
        for num in range(self.numParticles):
            selectPoint = random.random()
            accumulator = 0.0
            for i, p in enumerate(particles):
                accumulator += p.w
                if accumulator >= selectPoint:
                    break
            new_particles.append(particles[i])
        return new_particles
        '''

    def displayParticle(self, img, p, color):
        # print(p)
        x0 = max(0, round(p.x-0.5*p.width))
        y0 = max(0, round(p.y-0.5*p.height))
        x1 = x0+round(p.width)
        y1 = y0+round(p.height)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, 8, 0)

    def displayParticles(self, img, particles, color,param):
        num = len(particles)
        if param == SHOW_ALL:
            for i in range(num-1, -1, -1):
                self.discircles(img, particles[i], color)
        elif param == SHOW_SELECTED:
            self.displayParticle(img, particles[0], color)
        return img

    def discircles(self, img, p, color):
        cv2.circle(img, (round(p.x), round(p.y)),
                   round(100*p.w), (0, 255, 0), 2, 8, 0)


if __name__ == '__main__':
    #path = r'/home/xd/graduate/MOTDT/data/MOT16/train/MOT16-02/img1/000001.jpg'
    #img= cv2.imread(path)
    PFiltetr = ParticleFilter()
    root = r'D:\material\experiment\MOT16\train\MOT16-02\img1'
    files = os.listdir(root)
    files.sort()
    tlbr = [439, 444, 560, 720]
    for num, filename in enumerate(files):
        file = os.path.join(root, filename)
        img = cv2.imread(file)
        h, w, _ = img.shape
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if num == 0:
            '''初始化'''
            particles = PFiltetr.initiate(tlbr)
            objectHist = PFiltetr.colorHist(imgHSV, particles[0], h, w)
        else:
            '''粒子传播'''
            time1 = time.clock()
            particles = PFiltetr.transition(particles, w, h)
            time2 = time.clock()
            '''更新权重并归一化'''
            particles = PFiltetr.updateweight(particles,imgHSV,objectHist,h,w)
            time3 = time.clock()
            '''重采样'''
            particles = PFiltetr.resample(particles)
            time4 = time.clock()
        img = PFiltetr.displayParticles(img,particles,(0,0,255),SHOW_SELECTED)
        cv2.imshow('img', img)
        cv2.waitKey(1)
