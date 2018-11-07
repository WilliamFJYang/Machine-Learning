import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,1000,(10000,2))

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS
# Apply KMeans
ret,label,center = cv.kmeans(trainData.astype("float32"),4,None,criteria,10,flags)

A = trainData[label.ravel()== 0]
B = trainData[label.ravel()== 1]
C = trainData[label.ravel()== 2]
D = trainData[label.ravel()== 3]

print(trainData)
plt.scatter(A[:,0],A[:,1],20,'r','^')
plt.scatter(B[:,0],B[:,1],20,'b','o')
plt.scatter(C[:,0],C[:,1],20,'g','.')
plt.scatter(D[:,0],D[:,1],20,'k','x')

plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.show()
