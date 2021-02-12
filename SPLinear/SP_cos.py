#This code implements a variant of the signal perceptron using real sinousoids instead of analytic sinousoids
import random as rm
import numpy as np

m=8;  #Módule (base) Ej: si m=3, se genera 0,1,2,0,1,2,...
k=2; #Number of nested loops. That is, number of variables
aix=np.zeros([k]); #Array of indexes (to order them)
aiw=np.zeros([k]); #Array of indexes (to order them)
ni=m**k   #Number of Iterations
n=k  #No. of variables
nn=m**n #|m^k| domain space
nnn=m**nn #|Delta|=|m^m^k| function space
# Matrix for  
A=np.zeros([nn,nn]) 
divfrec=m-1
i=0; j=0
v=0; 
#Function for generating the matrix for m^k perceptron

for xi in range(0,ni,1):
	kx=xi;
	for xj in range(0,k,1): #Generamos los índices
		aix[xj]= int ( kx % m ); #Lo metemos en array 
		kx=int(kx/m); #siguientes índices
	#print("aix=",aix)
	j=0;
	#First Inner nested loop that generates all combinations of w for a signal
	for wi in range(0,ni,1):
		kw=wi;
		for wj in range(0,k,1): 
			aiw[wj]= int ( kw % m ) ; 
			kw=int(kw/m); 
			#print(i,j,A[i,j],"|",end='')
		exponente=0
		#Second Inner loop that  multiplies and sums
		for ii in range(0,k,1):
			exponente=exponente + aix[ii]*aiw[ii]
			exponente=int(exponente)
		#print("exponente=",exponente)
		exponente=np.pi*exponente/divfrec
		#print(exponente)
		#print(np.exp(exponente))
		A[i,j]=np.cos(exponente)
		#print(A[i,j])
		j=j+1
		#print("aiw=",aiw,"j=",j)
		#for aj in range(0,nc,1):
		#	print(i,j,A[i,j],"|",end='')
		#	print()
	i=i+1
print("A= \n",A)
#a1=A.astype(int) only uncomment this if you dont want the matrix to use the complex part
atrans= A.transpose()
print("A inverse= \n",atrans)
#h=np.matmul(A,atrans)
#print(h)
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

#Function that generates a random sample from the m^k function space
def rand_func(m,k):
	b=np.zeros([m**k])
	fun=rm.randint(0,m**m**k)
	r=fun
	for j in range(0,m**k):
		r=fun%m
		fun=int(fun/m)
		b[j]=r
	return b
#Function that obtains the alphas for n randomly sample functions from a m^k function space.
def solve_n_rand_func(n):
	for i in range(0,n):
		y=rand_func(m,k)
		X = np.linalg.inv(A).dot(y)
		print("Function ",i,"):",y,X)

#all_func();
solve_n_rand_func(10)

print("The function space size is: ",nnn)
print(" End of lpn")
