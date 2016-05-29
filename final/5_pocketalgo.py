#PocketAlgorithm
import matplotlib.pyplot as plt
from random import randint
import math as m
import numpy as np
unit_step = lambda x: 1 if x >= 0 else -1
rate=0.001
theta=0
filename = 'linear.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1]),float(t[2])] for t in temp]#type cast each point to float

labels =[ float(t[4]) for t in temp]


weights = []
finalsuccess=[]
finalweights=[]
finale=[]
misclassified=[]
for k  in range(0,4):
  	weights.append(randint(0,1))
# print "initial"
# print weights



iteration=0
def calOutput(theta,weights,x,y,z):
	sum = x*weights[0]+y*weights[1]+z*weights[2]+weights[3]
	return unit_step(sum)


def perceptron():"""Initial Weight vector of 1xd assumed to be 0/1,the algorithm breaks on iteration count 500,we have plotted the misclassified 
points against the iteration number"""
	iteration=0
	
	while True:
		
		globalError=0
		iteration +=1
		error_count=0
		success=0
		
		#print "hi"
		for i in range(0,len(dataPoints)):
			#print "iteration" + str(i)
			output= calOutput(theta,weights,dataPoints[i][0],dataPoints[i][1],dataPoints[i][2])
			
			localError = labels[i] -output
			if localError != 0:
				error_count += 1
								
				for k in range(3):
					weights[k] += rate*localError*dataPoints[i][k]

				weights[3]+=rate*localError
			else:
				success+=1
			

				#print "local " + str(localError)
				#globalError+= localError**2
			#print "global " + str(globalError)		
		#print weights	
		finalweights.append(list(weights))
		#print finalweights
		misclassified.append(error_count)
		finalsuccess.append(success)
		
		
		if (iteration==500):
			print "Number of Iterations " + str(iteration)
			print "Final Maximum classifications"
			print max(finalsuccess)
			print "finalweights"
			print finalweights[finalsuccess.index(max(finalsuccess))]
			finale.append(finalweights[finalsuccess.index(max(finalsuccess))])
			break



perceptron()


#print "Final weight vector " +str(weights)	
print "Equation of hyperplane " + str(finale[0][0])+"x + "+ str(finale[0][1])+"y +"+str(finale[0][2])+"z +"+str(finale[0][3])+" = 0"


count=0

y = np.array(misclassified)
N = len(y)
x = range(N)
plt.axis([0, 500, 0, 50])
plt.suptitle("# of Misclassifications vs Iterations")
plt.plot(x,y, '-', linewidth=0.5,color='black')
#plt.bar(x, y, width, color="blue")

plt.xlabel("Iterations")
plt.ylabel("# of Misclassifications")
plt.show()



