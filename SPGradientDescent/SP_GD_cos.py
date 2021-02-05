import numpy as np
import random as rm




def Signal_perceptron_gen(m,k):
	wki=[]
	aiw=np.zeros([k]);
	for i in range(0,m**k,1):
		kw=i;
		for j in range(0,k,1): 
			aiw[j]= int ( kw % m ); 
			kw=int(kw/m); 
		w=[]
		for l in aiw:
			w.append(l)
		wki.append(w)
	arrw = np.asarray(wki)
	print("frecuency matrix",arrw.shape)
	def signal_perceptron(alpha,x):
		y_pred = 0
		#print("x",x.shape)
		x = np.transpose(x)
		#print("x trans",x.shape)
		exp=np.dot(arrw,x)
		#print("exponent",exp.shape)
		o_sp=np.cos(np.pi*exp)
		#print("after exponential",o_sp)
		#print("theta vector",theta.shape)
		y_sp=np.dot(alpha,o_sp)
		#print("result",y_sp)
		return y_sp , o_sp
	return signal_perceptron
	
def gradientDescent(X, Y, Alpha, m,k,gamma=.1):
    SP = Signal_perceptron_gen(m,k)
    N=len(X)
    for i in range(0, 100):
        #print(X)
        hypothesis ,m_exp= SP(Alpha,X)
        loss = Y-hypothesis
        #print(loss,"\n",m_exp)
        gradient= -2/N*np.dot(loss,m_exp)
        #print(gradient)
        Alpha = Alpha - gamma * gradient
    return Alpha

def loss(y_label,y_pred):
	n=len(y_label)
	loss= (y_label-y_pred)**2
	loss= 1/n*(np.sum(loss))
	print ("real vs predicted",y_label,y_pred)
	print(loss)
	

def data_gen(m,k,y=0):
	alpha=np.zeros([m**k])
#Creating dataset:
	xki=[]
	aix=np.zeros([k]);
	for i in range(0,m**k,1):
		kx=i;
		for j in range(0,k,1): 
			aix[j]= int ( kx % m ); 
			kx=int(kx/m); 
		xt=[]
		for l in aix:
			xt.append(l)
		xki.append(xt)
	X = np.asarray(xki)
	#Boolean Funtion:
	if m==2 and y.any!=0:
		return X, y,alpha
	
	else:
	#Creating random function
		b=np.zeros([m**k])
		fun=rm.randint(0,m**m**k)
		r=fun
		for j in range(0,m**k):
			r=fun%m
			fun=int(fun/m)
			b[j]=r
		return X, b, alpha
		
gamma= np.array([0,1,1,0])
x,y,a=data_gen(2,2,gamma)	
print(y)
alpha_gradient=gradientDescent(x,y,a,2,2)
print(alpha_gradient)
SP=Signal_perceptron_gen(2,2)
hypothesis,x = SP(alpha_gradient,x)
loss(y,hypothesis)
