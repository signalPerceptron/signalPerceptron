import numpy as np
import random as rm
k=3
m=2
n=k  #No. of variables
nn=2**n #|m^k| domain space
nnn=2**nn #|Delta|=|m^m^k| function space

def walshMatrix(n):
	N=2**n
	H=np.zeros([N,N])
	for i in range(0,N):
		for j in range(0,N):
			l=0
			ij = i & j;
			while ij!=0: 
				ij=ij&(ij-1);
				l=l+1
			if(l%2==0):
				H[i,j] = 1;
			else:
				 H[i,j] = -1;
	return H

A=walshMatrix(n)

atrans= A.transpose()
print(atrans)

#Function for calculating all posible functions from a m^k function space (Warning: this could fail if the function space is to big since it will require lots of memory)
def all_func():
	b=np.zeros([nnn,nn])
	alfa=np.zeros([nnn,nn])
	#Function that obtains the parameters of all possible functions
	for i in range(0,nnn):
		ii=i; r=i;
		#base transformation function
		for j in range(0,nn):
			r=ii%m
			ii=int(ii/m)
			b[i,j]=r
		#print(b[i])
		X = np.linalg.inv(A).dot(b[i])
		alfa[i]=X
		print("Function ",i,"):",b[i],alfa[i])
		#print(b[i])
		#print("weight:",alfa[i])
	#print(b)
	#print(alfa)
#Function for calculating all posible functions from a m^k function space (Warning: this could fail if the function space is to big since it will require lots of memory)
def single_func():
	#Some boolean functions of different arity: 2^2,2^3,2^4,2^5
	b5=np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0]) #arity 5
	b4=np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0]) #arity 4
	b3=np.array([0,1,1,0,1,0,0,1]) #arity 3 XOR
	b2=np.array([0,1,1,0]) #arity 2 XOR
	#Some trilean functions of different arity: 3^2,3^3
	t3=np.array([0,1,2,1,1,2,0,2,1,0,1,1,0,1,0,0,1,1,1,1,2,0,2,1,0,1,1]) #arity 3 
	t2=np.array([0,1,2,1,1,2,0,2,1]) #arity 2 
	t2v1=np.array([1, 2, 2, 1, 1, 1, 1, 0, 1])
	#Some m^k functions:
	p2=np.array([0,1,2,3,1,2,4,2,1,2,3,1,3,3,0,0,4,1,2,1,3,0,4,1,0]) 
	h3=np.array([5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,5,2,3,1,3,3,5,0,4,1,2,5,3,0,4,1,0,0,1,2,5,1,2,4,2,1,2,5,4,3,4,2,0,1,4,2,5,3,4,4,3]) 
	o2=([0,1,2,3,1,2,5,2,1,6,3,1,5,3,7,0,4,1,2,1,3,0,7,1,7,0,1,2,3,1,2,5,2,1,6,3,1,5,3,7,0,4,1,2,1,3,0,7,1,7,1,3,4,7,6,1,2,3,7,0,2,5,3,7])
	X = np.linalg.inv(A).dot(o2)
	print("parameters for function ",o2,": \n",X)
	
all_func();
#single_func();
print("The function space size is: ",nnn)
print(" End of lpn")



