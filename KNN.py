import cv2
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 100 known/training data
trainData = np.random.randint(0,1000,(100,2)).astype(np.float32)

# Labels each one either A,B,C,D with numbers 0 to 3
responses = np.random.randint(0,4,(100,1)).astype(np.float32)

# Take A families and plot them
A = trainData[responses.ravel()==0]
plt.scatter(A[:,0],A[:,1],20,'r','^')

# Take B families and plot them
B = trainData[responses.ravel()==1]
plt.scatter(B[:,0],B[:,1],20,'b','s')
# Take C families and plot them
C = trainData[responses.ravel()==2]
plt.scatter(C[:,0],C[:,1],20,'k','*')
# Take D families and plot them
D = trainData[responses.ravel()==3]
plt.scatter(D[:,0],D[:,1],20,'y','8')
#Make the New data to predict by KNN
newcomer = np.random.randint(0,1000,(10,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],20,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 5)

print ("result: \n", results,"\n")
print ("neighbours: \n", neighbours,"\n")
print ("distance: \n", dist.astype("uint8"))

plt.show()
