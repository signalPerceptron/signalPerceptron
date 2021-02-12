import numpy as np
import random as rm
import matplotlib.pyplot as plt

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
	#print("frecuency matrix",arrw.shape)
	def signal_perceptron(alpha,x):
		y_pred = 0
		#print("x",x.shape)
		x = np.transpose(x)
		#print("x trans",x.shape)
		exp=np.dot(arrw,x)
		#print("exponent",exp.shape)
		o_sp=np.cos((np.pi/(m-1))*exp)
		#print("after exponential",o_sp)
		#print("theta vector",theta.shape)
		y_sp=np.dot(alpha,o_sp)
		#print("result",y_sp)
		return y_sp , o_sp
	return signal_perceptron
	
def gradientDescent(X, Y, Alpha, m,k,gamma=.1):
    history_train=[]
    SP = Signal_perceptron_gen(m,k)
    N=len(X)
    for i in range(0, 100):
        #print(X)
        hypothesis ,m_exp= SP(Alpha,X)
        loss1 = Y-hypothesis
        #print(loss,"\n",m_exp)
        gradient= -2/N*np.dot(loss1,m_exp)
        #print(gradient)
        Alpha = Alpha - gamma * gradient
        history_train.append([i,loss(Y,hypothesis)])
    return Alpha, history_train

def loss(y_label,y_pred):
	n=len(y_label)
	loss= (y_label-y_pred)**2
	loss= 1/n*(np.sum(loss))
	#print ("real vs predicted",y_label,y_pred)
	#print(loss)
	return loss

def data_gen(m,k,y=0):
	#Initializing weight vector
	alpha=np.ones([m**k])
	alpha1=np.zeros([m**k])
	alpha2=np.random.rand(1,m**k)
	alpha4= .5 * np.random.randn(1, m**k) 
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
	if m==2 and y!=0:
		return X, y,alpha
	
	else:
	#Creating random function
		b=np.zeros([m**k])
		fun=rm.randint(0,m**m**k)
		r=fun
		print(m)
		for j in range(0,m**k):
			r=fun%m
			fun=int(fun/m)
			b[j]=r
		return X, b, alpha4
		
#gamma1= [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
#gamma2=[[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],[1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],[2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2]]

def allfuncplot(m,k):
	for i in gamma2:
		x_axis=[]
		y_axis=[]
		x,y,a=data_gen(m,k,np.array(i))	
		print(a)
		alpha_gradient,hist_train=gradientDescent(x,y,a,m,k)
		for j in hist_train:
			x_axis.append(j[0])
			y_axis.append(j[1])
		a=" ".join(str(x) for x in i)
		print(a)
		plt.plot(x_axis,y_axis, label = a)
		plt.title('Training loss: Unary Trilean functions using Real-Signal Perceptron')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
	print(alpha_gradient)
	plt.show()

def rand_n_funcplot(n,m,k):
	for i in range(0,n):
		x_axis=[]
		y_axis=[]
		x,y,a=data_gen(m,k)	
		print(y)
		alpha_gradient,hist_train=gradientDescent(x,y,a,m,k)
		for j in hist_train:
			x_axis.append(j[0])
			y_axis.append(j[1])
		#a=" ".join(str(x) for x in y)
		#print(a)
		a= "Function:"+str(i)
		plt.plot(x_axis,y_axis, label = a)
		titles='Training loss of '+str(n)+' random: '+str(m)+'-values '+str(k)+'-ary  functions using Real-Signal Perceptron'
		plt.title(titles)
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
	print(alpha_gradient)
	plt.show()

#allfuncplot(3,1)
rand_n_funcplot(10,8,2)
#SP=Signal_perceptron_gen(2,2)
#hypothesis,x = SP(alpha_gradient,x)
#loss(y,hypothesis)


