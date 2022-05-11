import copy
from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import os



np.set_printoptions(suppress=True)
indexs = [3, 9, 13, 16, 19, 6]
MATRIX_B1 = np.asarray([[ 451.55322537,  -23.54276529,  492.4057058,  1045.79364383],
 [ 277.09854842,  -14.74778966,  331.5450658,   619.33617748],
 [   0.43138119,  -0.02248432,    0.46920329,   1        ]])
MATRIX_B3 = np.asarray([[223.26089591, -11.73461386, 240.08486431, 519.62369615],
 [315.21511932, -16.89727167, 378.59634489, 703.31988798],
 [  0.43209616,  -0.02277389,   0.47207236,   1       ]])

'''
MATRIX_B1 = np.asarray([[ 849.77075856,  -69.56167559, 2146.16636095, 1046.71992891],
 [467.02973169,  -42.48548829, 1410.70732728,  400.2542614],
 [0.8110194,    -0.06639029,    2.04698316,    1]]
) 
'''

'''读取飞机的3D坐标'''
def read3DFile(filePath):
    threeDloc = []
    with open(filePath) as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            x,y,z = line[-3],line[-2],line[-1]
            if '.' in x:
                x,y,z = float(x),float(y),float(z)
                threeDloc.append([x,y,z])
    return threeDloc

'''读取飞机的2D坐标'''
def read2DFile(filePath):
    twoDloc = []
    with open(filePath) as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            x,y = line[-2],line[-1]
            if '.' in x:
                x,y = float(x),float(y)
                twoDloc.append([x,y])
    return twoDloc  


'''获得关于2D的转换结果'''
def to2D(transMatrix,thrD):
    thrData = copy.deepcopy(thrD)
    testData = []
    for da in thrData:
        da.append(1)
        #print('da=',da)
        da = np.asarray(da).reshape((4,1))
        res = np.dot(transMatrix,da)
        res = res[:,0]
        res = res/res[2]
        res = res[:2]
        testData.append(res)
    return testData

'''计算误差'''
def getErr(testData,realData):

    arr = []
    num = len(testData)
    for i in range(num):
        d = abs(realData[i][0]-testData[i][0])+abs(realData[i][1]-testData[i][1])
        arr.append(d)
    arr = np.asarray(arr).astype(np.int32)
    return arr

'绘制误差曲线'
def drawErr(realB1,testB1,realB3,testB3,saveRoot):
    arr1 = getErr(testB1,realB1)
    arr3 = getErr(testB3,realB3)
    plt.plot(arr1,color='r',label='B1')
    plt.plot(arr3,color='b',label='B3')
    plt.title('err')
    plt.legend()
    #plt.show()
    plt.savefig(saveRoot)
    plt.close()

'''绘制真实和映射的坐标点'''
def drawDot(testData,realData,type,saveRoot):
    num = len(testData)
    #print('testData=',testData)
    #print('realData=',realData)
    for i in range(num):
        test = plt.scatter(testData[i][0],testData[i][1],color='r')
        real = plt.scatter(realData[i][0],realData[i][1],color='b')
    plt.title(type)
    plt.legend((test,real),('map dot','real dot'))
    #plt.show()
    plt.savefig(saveRoot)
    plt.close()

def run(pathB1,pathB3,pathB1B3,indexs,saveroot = r'E:\work\filter\result'):
    realB1 = read2DFile(pathB1)
    realB3 = read2DFile(pathB3)
    realB1B3 = read3DFile(pathB1B3)

    #print('realB1B3=',realB1B3)
    testB1 = to2D(MATRIX_B1,realB1B3)
    testB3 = to2D(MATRIX_B3,realB1B3)
    filename = ''
    for i in indexs:
        filename += str(i)+'_'
    filenameErr =  filename+'err.jpg'
    filenameB1 = filename+'B1.jpg'
    filenameB3 = filename+'B3.jpg'
    savePathB1 = os.path.join(saveroot,filenameB1)
    savePathB3 = os.path.join(saveroot,filenameB3)
    savePathErr = os.path.join(saveroot,filenameErr)
    drawDot(testB3,realB3,'B3',savePathB3)
    drawDot(testB1,realB1,'B1',savePathB1)
    drawErr(realB1,testB1,realB3,testB3,savePathErr)
    


threeDfilepath = r'E:\work\filter\airplane\b1b3.txt'
twoDfilepath_B1 = r'E:\work\filter\airplane\data_b1.txt'
twoDfilepath_B3 = r'E:\work\filter\airplane\data_b3.txt'
resultRoot = r'E:\work\filter\result'
run(twoDfilepath_B1,twoDfilepath_B3,threeDfilepath,indexs,resultRoot)
    