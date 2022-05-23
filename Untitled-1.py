from cProfile import label
from operator import inv
from pickletools import optimize
from tkinter.ttk import Scale
from turtle import color
import numpy as np
from numpy.linalg import norm ,det ,inv,qr,eig
from numpy import arccos , dot 
import matplotlib.pyplot as plt

# row=np.array([[1,2,3]])
# col=np.array([[1],[2],[3]])
# print(row.shape)

# new_vex = row
# norm_1= norm(new_vex,1)
# print(norm_1)

# v=np.array([[1,2,3]])
# w=np.array([[4,5,6]])
# theta = arccos(dot(v,w.T)/(norm(v)*norm(w)))
# print(theta)
# print(np.cross(v,w))

# a=np.array([[1,8],[4,6],[7,1]])
# b=np.array([[5,2,2],[9,7,1]])
# x=np.dot(a,b)
# print(x)

# m=np.array([[1,2,3],
#             [4,9,7], 
#             [9,7,6]])
# print (det(m))
# I=np.eye(3)
# print(np.dot(I,m))
# print(inv(m))

# m=[[8,3,-3],[-2,-8,5],[3,5,10]]
# diag=np.diag(np.abs (m))
# offdiag=np.sum(np.abs(m),axis=1)-diag
# if np.all(diag>offdiag):
#     print("dom")
# else :
#     print("not")
# x1=0
# x2=0
# x3=0
# epsilon = .01
# converged = False
# x_old = np.array([x1,x2,x3])
# print ("iteration result")
# print ("k,  x1,  x2,  x3")
# for k in range (1,50):
#     x1= (14-3*x2+3*x3)/8
#     x2=(5+2*x1-5*x3)/(-8)
#     x3=(-8-3*x1-5*x2)/(-5)
#     x=np.array([x1,x2,x3])
#     dx=np.sqrt(np.dot(x-x_old,x-x_old))
#     print(k, x1,x2,x3)
#     if dx<epsilon:
#         converged=True
#         print("con")
#         break
#     x_old=x
# if not converged:
#     print("Not converged, increase the number of iterations")

# a=
# y=
#x=np.linalg.solve(a,y)

# from scipy.linalg import lu
# a = np.array([[4, 3, -5],[-2,-4,5],[8,8,0]])
# p,l,u = lu(a)
# print("p",p) 
# print("l",l)
# print("u:",u)
# print("lu:",np.dot(l,u))

# def plot_vec(x,y,xlim,ylim):
#     plt.figure(figsize=(10,6))
#     plt.quiver(0,0,x[0],x[1],color="r", angles="xy",scale_units="xy",scale =1,label="original")
#     plt.quiver(0,0,y[0],y[1],color="g", angles="xy",scale_units="xy",scale =1,label="transform")
#     plt.xlim(xlim)
#     plt.ylim(ylim)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.show()
# plot_vec(x,y,(0,3),(0,2))

#power metod largest
# def normalize(x): 
#     max=abs(x).max()
#     xnew=x/x.max()
#     return max , xnew
# x=np.array([1,1])
# a=np.array([[0,2],[2,3]])
# ainv=inv(a)####smallest 
# for i in range (8):
#     x=np.dot(x,a)
#     y,x=normalize(x)
# print(x,'evec',y)

#qrlinalg
# a=np.array([0,2],[2,3])
# q,r=qr(a)
# print (q,r)
# b=np.dot(q,r)
# print(b)

# a=np.array([[0,2],[2,3]])
# p_itration=[1,5,10,20]
# for i in range(20):
#     q ,r = qr(a)
#     a=np.dot(q,r)
#     x=np.dot(r,q)
#     if i+1 in p_itration :
#         print ("itration ",{i+1})
#         print(a)

#eiglian
# a=np.array([[0,2],[2,3]])
# w,v=eig(a)
# print(w,v)

#least squares regression
# x=np.linspace(0,1,101)
#print(x)
# y= 1 + x + x*np.random.random(len(x))
#print(y)
# A=np.vstack([x,np.ones(len(x))]).T
# y=y[:,np.newaxis]
# pseudo=np.linalg.pinv(A)
# alpha=pseudo.dot(y)
#alpha=np.linalg.lstsq(a,y,rcond=none)[0]
#print(alpha)
# plt.style.use("seaborn-poster")
# plt.figure(figsize=(10,6))
# plt.plot(x,y,"b")
# plt.plot(x, alpha[0] * x + alpha[1],'r')
# plt.show()

#USING OPTIMIZE.CURVE_FIT FROM SCIPY
# x=np.linspace(0,1,101)
# y= 1 + x + x*np.random.random(len(x))
# def func(x,a,b):
#     y=a*x+b
#     return y
# alpha=optimize.curve_fit(func,xdata=x,ydata=y)[0]
# print(alpha)

#non linear least square 
# x=np.linspace(0,10,101) 
# y=0.1 * np.exp(0.3*x)+0.1*np.random.random(len(x))
# A=np.vstack([x,np.ones(len(x))]).T
# beta ,logalpa=np.linalg.lstsq(A,np.log(y),rcond=None)[0]
# alpha=np.exp(logalpa)
# plt.style.use("seaborn-poster")
# plt.figure(figsize=(10,6))
# plt.plot(x,y,"b")
# plt.plot(x,alpha*np.exp(beta*x),'r')
# plt.show()

# #plot the real function:
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use("seaborn-poster")
# x = np.linspace(0, 10, 101)
# y = 0.1 * np.exp(0.3*x) + 0.1 * np.random.random(len(x))
# plt.figure(figsize=(10, 8))
# plt.plot(x, y, "b")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# #LOG TRICKS FOR EXPONENTIAL FUNCTIONS:
# import numpy as np
# from scipy import optimize
# import matplotlib.pyplot as plt
# plt.style.use("seaborn-poster")
# x = np.linspace(0, 10, 101)
# y = 0.1*np.exp(0.3*x) + 0.1*np.random.random(len(x))
# plt.figure(figsize = (10,8))
# plt.plot(x, y, "b.")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
# A = np.vstack([x, np.ones(len(x))]).T
# beta, log_alpha = np.linalg.lstsq(A, np.log(y), rcond = None)[0]
# alpha = np.exp(log_alpha)
# print(f"alpha={alpha}, beta={beta}")
# plt.figure(figsize = (10,8))
# plt.plot(x, y, "b.")
# plt.plot(x, alpha*np.exp(beta*x), "r")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# #POLYNOMIAL REGRESSION:
# import numpy as np
# from scipy import optimize
# import matplotlib.pyplot as plt
# plt.style.use("seaborn-poster")
# x_d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# y_d=np.array([0,0.8,0.9,0.1,-0.6,-0.8,-1,-0.9,-0.4])
# plt.figure(figsize = (12, 8))
# for i in range(1, 7):
#     y_est = np.polyfit(x_d, y_d, i)
#     plt.subplot(2,3,i)
#     plt.plot(x_d, y_d, "o")
#     plt.plot(x_d, np.polyval(y_est, x_d))
#     plt.title(f"Polynomial order {i}")
# plt.tight_layout()
# plt.show()

# #USING OPTIMIZE.CURVE_FIT FROM SCIPY:
# import numpy as np
# from scipy import optimize
# import matplotlib.pyplot as plt
# plt.style.use("seaborn-poster")
# x = np.linspace(0, 10, 101)
# y = 0.1*np.exp(0.3*x) + 0.1*np.random.random(len(x))
# def func(x, a, b):
#     y = a*np.exp(b*x)
#     return y
# alpha, beta = optimize.curve_fit(func, xdata = x, ydata = y)[0]
# print(f"alpha={alpha}, beta={beta}")
# plt.figure(figsize = (10,8))
# plt.plot(x, y, "b.")
# plt.plot(x, alpha*np.exp(beta*x), "r")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
