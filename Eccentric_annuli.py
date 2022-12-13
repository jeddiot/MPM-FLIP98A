from math import cos
from os import write
import numpy as np
import matplotlib.pyplot as plt
import xlwt

book = xlwt.Workbook(encoding='utf-8')
filename = 'HW2_simularity.xls'
data = book.add_sheet('1',cell_overwrite_ok=True)
data.write(0,0,'normalized position')
data.write(0,1,'normalized velocity')

fig = plt.figure()
ax = plt.axes()
ax.set_aspect('equal', adjustable='box')

# This code is based on "An Analysis of Fully Developed Flow in an Eccentric Laminar Annulus", WILLIAM T. SNYDER, 1965
# maximum numbers of n used when do the summention
sumnum = 5

# geometry propertis
r1 = 0.02
r2 = 0.05
e = 0.025

X = []
U = []

gamma = r1/r2 # eq(4d)
phi = e/(r2-r1) # eq(4e)

c1 = 1+phi**2
c2 = 1-phi**2

# resolution of output graphic
angle_divide = 360
rad_divide = 30
test_angle = [0,45,180,315]

a = np.arccosh((gamma*c1+c2)/gamma/2/phi)
b = np.arccosh((gamma*c2+c1)/2/phi)
c = r1*np.sinh(a)

# cotangent hyporbolic
def ctnh(k):
    return 1/np.tanh(k)

# be ready to sum up first few term of a infinite series  
# return a list indicate the form at n=1,2,3... 
def sum_coefficient_of_n(k):
    i = np.ndarray([sumnum-1])
    for p in range(1,sumnum):
        i[p-1] = p*k
    return i

# shorten calulation
ctnha = ctnh(a)
ctnhb = ctnh(b)
ctnha_b = ctnha-ctnhb

F = (a*ctnhb-b*ctnha)/2/(a-b) # eq(6b)
E = (ctnha_b)/2/(a-b) # eq(6c)

# shorten calulation
ea = np.exp(sum_coefficient_of_n(2*a))
eb = np.exp(sum_coefficient_of_n(2*b))
ea_b = ea-eb

A = ctnha_b/ea_b # eq(6d)
B = (ea*ctnhb-eb*ctnha)/ea_b # eq(6e)

for i in range(1,angle_divide):
    xi = i*np.pi/(angle_divide/2)
    for j in range(1,rad_divide+1):
        eta = j*(a-b)/rad_divide+b
        x = c*np.sinh(eta)/(np.cosh(eta)-cos(xi)) # eq(3a)
        y = c*np.sin(xi)/(np.cosh(eta)-cos(xi)) # eq(3b)
        v = F + E*eta - ctnh(eta)/2 + np.sum((A*np.exp(sum_coefficient_of_n(eta))+(B-ctnh(eta))*np.exp(sum_coefficient_of_n(-eta)))*np.cos(sum_coefficient_of_n(xi))) # eq(6a)
        u = -v*(c**2)/(1.7894*10**(-5))*((0.0125-0.013)/0.2) # eq(5b): u = -v*(c^2/mu)*(dP/dz), for Re 100, dp/dz = (0.0125-0.013)/0.2.
        X.append(x)
        U.append(u)

        # define color with range of velocity
        img = ax.scatter(x, y, s=5, c = u, cmap='jet', vmin=0, vmax=0.04605) # Re=100
min_x = min(X)
min_u = min(U)
delt_x = max(X)-min_x

for j in range(rad_divide*3):
    data.write(j+1,0,(X[j]-min_x)/delt_x) 
    data.write(j+1,1,(U[j]-min_u)/delt_x)
book.save(filename)

plt.title('Analytic Sol. of Velocity')
plt.colorbar(img)   # apply color bar
plt.show()
