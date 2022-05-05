

try:
    import cv2
    import scipy
    from scipy import stats
    import scipy.optimize as sco
except :
    print("failed to import opencv, please install opencv-python by 'pip install opencv-python'")
    pass

object_points = [
        [-3.112418, 0.963687,0.773610,1],
        [-3.113543, 0.934069,0.772541,1],
        [-3.114047, 0.888104,0.771655,1],
        [-3.114866, 0.822488,0.771336,1],
        [-3.119577, 0.704497,0.771588,1],
        [-3.122606, 0.637091,0.770551,1]
    ]

image_points = [
        [516.750,722.250,1],
        [516.250,732.000,1],
        [514.250,748.000,1],
        [512.000,770.750,1],
        [507.000,813.000,1],
        [504.250,838.250,1]
    ]
image_size = (3840,2160)
# =========================================================== #
import numpy as np
np.set_printoptions(suppress=True)


if len(object_points) != len(image_points):
    print("object points must have same number with image points. object points :%d, image points %d"%(len(object_points),len(image_points)))
objPoints = np.array(object_points,dtype='float64')
objPoints.reshape((-1,3))
#objPoints = objPoints[np.newaxis]

imgPoints = np.array(image_points,dtype='float64')
imgPoints.reshape((-1,2))
#imgPoints = imgPoints[np.newaxis]

#M = imgPoints * objPoints.transpose()
X = objPoints
XT = objPoints.transpose()
XTX = np.dot( XT,objPoints)
XXT = X.dot(XT)
XTXR = np.linalg.inv( XTX)

M = np.dot(XTXR,objPoints.transpose()).dot(imgPoints)
print("M matrix:(Direct Calculated)\n",M.transpose())

P2 = objPoints.dot(M)
print("P", imgPoints)
print("P2", P2)

print("average error:", np.sum(np.sqrt( np.sum((imgPoints - P2)**2,axis=1)))/len(object_points))
#def target_func(x, P, X):
#    M = np.c_[x.reshape((4,2)),np.zeros(4)]
#    M[3,2] = 1.0
#    loss = np.log(np.sum(np.sqrt(np.sum((X.dot(M) - P)**2,axis=1)))/len(object_points))
#    return loss
#print("Start min-loss optmization")
#opt = sco.minimize(fun = target_func,x0 = M[:,0:2],args=(imgPoints, objPoints),tol=1e-3, method='BFGS',bounds=(None, None),options={'maxiter':1000,"disp": True,'gtol': 1e-02,})
#M = np.c_[opt.x.reshape((4,2)),np.zeros(4)]
#M[3,2] = 1.0
#print("optimize finished", opt)
#print("Optimized M matrix:")
#print(M.transpose(),"\naverage error\n", np.exp(opt.fun))
#P2 = objPoints.dot(M)
#print("P", imgPoints)
#print("P2", P2)
all_params = []
for i in range(len(image_points)):
    m1 = objPoints[i]
    m2 = np.array([
            [-1, 0, imgPoints[i,0]],
            [0 ,-1, imgPoints[i,1]]
        ])
    param = m1.reshape((-1,1)).dot(m2[0].reshape((1,-1)))
    param = param.transpose().flatten()
    all_params.append(param)
    param = m1.reshape((-1,1)).dot(m2[1].reshape((1,-1)))
    param = param.transpose().flatten()
    all_params.append(param)
all_params = np.array(all_params)
a = all_params[:-1,0:-1]
b = -all_params[:-1,-1]
from numpy.linalg import lstsq,solve
from numpy.linalg import matrix_rank,det
#print("parameter rank:", matrix_rank(a), " Det:", det(all_params))
#b = np.zeros((all_params.shape[0],1))
print("calculate matrix by first 11 rows. start.")
x = solve(a,b)
print(np.allclose(a.dot(x),b))
#x=lstsq(a,b)
print(x)

final_matrix = np.append(x,1.0)
final_matrix = final_matrix.reshape((3,4))
print("final matrix**:",final_matrix)
print("final matrix:", final_matrix.reshape((4,3)))
P2 = objPoints.dot(final_matrix.transpose())
P2 = P2/P2[:,-1:np.newaxis]
print("P", imgPoints)
print("P2", P2)

print("average error:", np.sum(np.sqrt( np.sum((imgPoints - P2)**2,axis=1)))/len(object_points))

print("calculate matrix by minimal error. start")
a = all_params[:,0:-1]
b = -all_params[:,-1]
x = lstsq(a,b)
#x=lstsq(a,b)
print(x[0])

final_matrix = np.append(x[0],1.0)
final_matrix = final_matrix.reshape((3,4))
print("final matrix***:",final_matrix)
print("final matrix:", final_matrix.reshape((4,3)))
P2 = objPoints.dot(final_matrix.transpose())
P2 = P2/P2[:,-1:np.newaxis]
print("P", imgPoints)
print("P2", P2)
error = np.sum(np.sqrt( np.sum((imgPoints - P2)**2,axis=1)))/len(object_points)
print("average error:",error)

print("\n\n ====== final matrix ====== \n", final_matrix.reshape((4,3)))
print("\n\n ====== final error  ====== \n", error)
