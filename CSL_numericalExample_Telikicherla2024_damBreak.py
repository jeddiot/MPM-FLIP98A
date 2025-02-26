import os
import time
import numpy as np
import pandas as pd
from pyevtk.hl import pointsToVTK
import taichi as ti
import colorama
import sys

time0 = time.time()
ti.init(arch=ti.gpu, default_ip = ti.i32, default_fp = ti.f64, device_memory_GB=3.0)

#-----------switches-----------#
isFBar = False # F-Bar pressure stabilization
# isDivvBar = False # True: use $\boldsymbol{\nabla} \cdot \boldsymbol{v}_0$; false: use $\boldsymbol{\nabla} \cdot \boldsymbol{v}_p$
isInterTimeStepDivv, deltaSL = False, 1
isPenaltyBC_elseBruteforceBC, betaNor = False, 1e6 # penalty
isBsplineKernel_elseTent = True
isMixedFormulation_elsePointwise = True
isAPIC_elsePIC = 1
isCSL_elseMPM, gammaNewmark, betaNewmark = True, 0.5, 0.25 # isCSLFLIPscheme2== False
eta_v, eta_u, eta_p = 0, 0, 0 # eta_v: how much FLIP in v, similar expression for u and p.
ifAV, c_artificial, c_L, c_Q = 0, 50, 0.12, 2.0 # artificial viscosity (Telikicherla et al. 2024)
# n_const = 7

omega = 1
visc = 1e-3 # 0 # 1.01e-3 # [Pa s], 0.001
# nu = 0.4999 # 4.999e-1 # [unitless], Poisson's ratio
rho = 997.5 # 997.5 # 1e3 # 997.5 # [kg m^{-3}], Particle density
kappa = 2e6 # 2e9*rho/1e3 # E/2/(1 - nu) # [Pa], bulk modulus
a_g = -9.81 # [m s^{-2}], a_g (should be negative)
# E, G = K * 2 * (1 - nu), K * (1 - nu)/(1 + nu) # [Pa] Young's modulus, [Pa] Shear modulus
# mu, lambd = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
#-----------numericalParameter-----------#
dim, sizeScale = 2, 1 # 6*1.05714285714 # 1
epsilon = 1e-15 # Numerical tolerance
simTime, timeTotal, dt = 3, 0e-15, 1e-6 # total simulation time, simulation time initialize, timestep

def format_with_exp(value):
    return f"{value:.2e}"

def append_params(filepath, vtkpath, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            suffix = f"_{key}{format_with_exp(value)}"
            filepath += suffix
            vtkpath += suffix
    return filepath, vtkpath

# Initialize the base filepath and vtkpath based on the condition
base_str = "CSL" if isCSL_elseMPM else "MPM"
filepath = f"mov_damBreak_Telikicherla2024_{base_str}_etav{eta_v}_etau{eta_u}_etap{eta_p}"
vtkpath = f"vtk_damBreak_Telikicherla2024_{base_str}_etav{eta_v}_etau{eta_u}_etap{eta_p}"

# Conditionally append parameters
# if isCSL_elseMPM:
#     filepath, vtkpath = append_params(filepath, vtkpath, gammaSL=gammaNewmark, betaSL=betaNewmark)

suffix = "_mixed" if isMixedFormulation_elsePointwise else "_ptwise"
filepath += suffix
vtkpath += suffix

if isPenaltyBC_elseBruteforceBC:
    filepath, vtkpath = append_params(filepath, vtkpath, betaNor=betaNor)

if isInterTimeStepDivv:
    filepath, vtkpath = append_params(filepath, vtkpath, deltaSL=deltaSL)

filepath, vtkpath = append_params(filepath, vtkpath, dt=dt)

np_x = 65
np_y = 65*2
num_p = np_x * np_y

len_domain = float(0.4375/sizeScale) # [m]
W_fluid = float(0.057/sizeScale) # [m] # Width of Liquid square (true dimension)
H_fluid = float(0.114/sizeScale) # [m] # Height of Liquid square (true dimension)
volume0_p = W_fluid*H_fluid/num_p

num_cell = int(100 + 4) # num_g+2+2 by num_g+2+2 including the boundaries
num_g = int(num_cell + 1) 
dx = len_domain  / float(num_cell - 4) # len_domain / float(num_g - 1)
inv_dx = 1 / dx

aNorm = 1.5 # RK normalized Kernel function support size
a = aNorm * dx # RK Kernel function support size
nodeNum = int(a*inv_dx*2 + epsilon) # (max) Number of 1D grid nodes in the support of each particle
shift = float(a*inv_dx - 1.0) # set as 0.5, used to find "base"

fb = ti.Vector.field(2, dtype=ti.f64, shape=()) # gravity
fb[None] = [0, W_fluid*H_fluid*rho*a_g] # [m kg s^{-2}], Set initial gravity direction to -y

beta = betaNor * rho * dx**2 # volume0_p # 1e30 # Penalty parameter on EBC

xtdt_p = ti.Vector.field(2, dtype=ti.f64, shape=num_p) # Particle position
vtdt_p = ti.Vector.field(2, dtype=ti.f64, shape=num_p) # Particle velocity
Lt_p = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p) # Velocity gradient (APIC)
Ft_p = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p) # Deformation gradient
sigma = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p) # Particle stress
atdt_p = ti.Vector.field(2, dtype=ti.f64, shape=num_p)
# if isCSLFLIPscheme2:
at_p = ti.Vector.field(2, dtype=float, shape=num_p)
utdt_p = ti.Vector.field(2, dtype=ti.f64, shape=num_p)
Delta_utdt_p = ti.Vector.field(2, dtype=ti.f64, shape=num_p)
material = ti.field(dtype=int, shape=num_p) # Material id
volumet_p = ti.field(dtype=ti.f64, shape=num_p) # Particle volume
mt_p = ti.field(dtype=ti.f64, shape=num_p)

mt_I = ti.Matrix.field(2, 2, dtype=ti.f64, shape=(num_g, num_g)) 
volumet_I = ti.field(dtype=ti.f64, shape=(num_g, num_g))
ptdt_I = ti.field(dtype=ti.f64, shape=(num_g, num_g))
pt_I = ti.field(dtype=ti.f64, shape=(num_g, num_g))
ft_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
vtdt_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
vt_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
atdt_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
at_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
Delta_utdt_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
Delta_ut_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
utdt_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))
ut_I = ti.Vector.field(2, dtype=float, shape=(num_g, num_g))

volume0_0 = ti.field(dtype=ti.f64, shape=(num_g - 1, num_g - 1)) # Pressure stabilization (F-bar)
volumet_0 = ti.field(dtype=ti.f64, shape=(num_g - 1, num_g - 1)) # Pressure stabilization (F-bar)
cell = ti.field(dtype=ti.f64, shape=(num_g - 1, num_g - 1)) # Pressure stabilization (F-bar)

detF = ti.field(dtype=ti.f64, shape=num_p) # determinant of F (deformation gradient)

PartitionOfUnity = ti.field(dtype=ti.f64, shape=num_p) # Check for each particle (POU)
Cons = ti.field(dtype=ti.f64, shape=num_p) # Check for each particle (consistency)
Cons_dx = ti.field(dtype=ti.f64, shape=num_p) # Check for each particle (gradient consistency)
Cons_dy = ti.field(dtype=ti.f64, shape=num_p) # Check for each particle (gradient consistency)

ptdt_p = ti.field(dtype=ti.f64, shape=num_p)
pt_p = ti.field(dtype=ti.f64, shape=num_p)
divvt_p = ti.field(dtype=ti.f64, shape=num_p) # \boldsymbol{nabla} \cdot \boldsymbol{v}_p^{t}
rho_p = ti.field(dtype=ti.f64, shape=num_p)

# -----------$div(\boldsymbol{v}_p^t)$-projection method
# divvt_0_numerator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
# divvt_0_denominator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
# divvt_0 = ti.field(dtype=ti.f64, shape=(num_g, num_g))

# -----------pBar method
pBar_numerator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
pBar_denominator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
pBar = ti.field(dtype=ti.f64, shape=(num_g, num_g))

# --------------------penaltyMethod
x_L_left = ti.Vector.field(2, dtype=ti.f64, shape=num_cell-4)
x_L_right = ti.Vector.field(2, dtype=ti.f64, shape=num_cell-4)
x_L_bot = ti.Vector.field(2, dtype=ti.f64, shape=num_cell-4)
x_L_top = ti.Vector.field(2, dtype=ti.f64, shape=num_cell-4)

@ti.func 
def getRK(xp, base, a): 
    phiMat = ti.Matrix.zero(ti.f64,3,3)                  # Initialize the matrix of phi values for each surrounding grid node at the current particle location
    dphiXMat = ti.Matrix.zero(ti.f64,3,3)                # Initialize the matrix of dphi/dx values for each surrounding grid node at the current particle location
    dphiYMat = ti.Matrix.zero(ti.f64,3,3)                # Initialize the matrix of dphi/dy values for each surrounding grid node at the current particle location
    M = ti.Matrix.zero(ti.f64,3,3)                       # Initialize the moment matrix 
    w = ti.Matrix.zero(ti.f64, nodeNum, dim)             # Initialize the kernel function vector for weights at each grid node for each coordinate direction (1D)
    weight = ti.Matrix.zero(ti.f64, nodeNum, nodeNum)    # Kernel weights in 2D 
    for i, d in ti.static(ti.ndrange(nodeNum, dim)):
        gridNode = float( i + base[d] ) * dx            # Current grid node location
        if gridNode >= 0:
            z = abs( xp[d] - float(gridNode)) / a       # Normalized distance from particle to grid node
            if isBsplineKernel_elseTent:
                if 0 <= z and z < 0.5:                      # Cubic b-spline kernel functions
                    w[i,d] = 2/3 - 4*z**2 + 4*z**3 
                elif 1/2 <= z and z < 1:
                    w[i,d] = 4/3 - 4*z + 4*z**2 - (4/3)*z**3
            else:
                if 0 <= z and z < 1:
                    w[i,d] = z - 1
                elif 1 <= z: 
                    w[i,d] = 0 
    for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)): 
        gridNode = [float(i+base[0]) * dx, float(j+base[1]) * dx] # Current grid node location
        if gridNode[0] >= 0 and gridNode[1] >= 0: 
            weight[i,j] = w[i,0] * w[j,1] # Define kernel function weights in 2D
            Pxi = (ti.Vector([1.0, xp[0] - gridNode[0], xp[1] - gridNode[1]])) # Define P(xi - xp)
            if weight[i,j] != 0:
                M += weight[i,j] * Pxi @ Pxi.transpose() # Define the moment matrix   

    # M_inv = M.inverse()
    M_inv = (M + epsilon * ti.Matrix.identity(ti.f64, 3)).inverse()

    for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)): # Loop over neighboring grid nodes                    
        gridNode = [float(i+base[0]) * dx, float(j+base[1]) * dx] # Current grid node location
        if weight[i,j] != 0:
            if gridNode[0] >= 0 and gridNode[1] >= 0:
                Pxi = ti.Vector([1.0, xp[0] - gridNode[0], xp[1] - gridNode[1]])  # Define P(xi - xp)
                Pxp = ti.Vector([1.0, xp[0] - xp[0], xp[1] - xp[1]])
                dPxpX = ti.Vector([0.0, -1.0, 0.0])
                dPxpY = ti.Vector([0.0, 0.0, -1.0])

                phi = weight[i,j] * (Pxp.transpose() @ M_inv @ Pxi)                   # Define phi
                dphi_x1 = weight[i,j] * (dPxpX.transpose() @ M_inv @ Pxi)             # Define dphi/dx
                dphi_x2 = weight[i,j] * (dPxpY.transpose() @ M_inv @ Pxi)             # Define dphi/dy
                
                phiMat[i,j] = phi[0]
                dphiXMat[i,j] = dphi_x1[0]
                dphiYMat[i,j] = dphi_x2[0]

    return phiMat, dphiXMat, dphiYMat

@ti.func 
def penaltybc(): 
    for k in x_L_left:
        base = (x_L_left[k] * inv_dx - shift).cast(int)
        Psi_I, _, _ = getRK(x_L_left[k], base, a)
        s1 = 1
        s2 = 0
        S = ti.Matrix([[s1, 0], [0, s2]])
        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
            offset = ti.Vector([i, j])
            if float(base[0] + offset[0]) >= 2: # <= 2:
                mt_I[base + offset] += dx * beta * Psi_I[i,j] * S
            if float(base[0] + offset[0]) < 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * S

    for k in x_L_right:
        base = (x_L_right[k] * inv_dx - shift).cast(int)
        Psi_I, _, _ = getRK(x_L_right[k], base, a)
        s1 = 1
        s2 = 0
        S = ti.Matrix([[s1, 0], [0, s2]])
        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
            offset = ti.Vector([i, j])
            if float(base[0] + offset[0]) <= num_cell - 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * Psi_I[i,j] * S
            if float(base[0] + offset[0]) > num_cell - 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * S

    for k in x_L_bot:
        base = (x_L_bot[k] * inv_dx - shift).cast(int)
        Psi_I, _, _ = getRK(x_L_bot[k], base, a)
        s1 = 0
        s2 = 1
        S = ti.Matrix([[s1, 0], [0, s2]])
        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
            offset = ti.Vector([i, j])
            if float(base[1] + offset[1]) >= 2: # <= 2:
                mt_I[base + offset] += dx * beta * Psi_I[i,j] * S
            if float(base[1] + offset[1]) < 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * S

    for k in x_L_top:
        base = (x_L_top[k] * inv_dx - shift).cast(int)
        Psi_I, _, _ = getRK(x_L_top[k], base, a)
        s1 = 0
        s2 = 1
        S = ti.Matrix([[s1, 0], [0, s2]])
        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
            offset = ti.Vector([i, j])
            if float(base[1] + offset[1]) <= num_cell - 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * Psi_I[i,j] * S
            if float(base[1] + offset[1]) > num_cell - 2: # >= num_cell - 2:
                mt_I[base + offset] += dx * beta * S

@ti.kernel
def substep():
    for i, j in mt_I:
        mt_I[i, j] = [[0, 0],[0, 0]]
        volumet_I[i, j] = 0

        ptdt_I[i, j] = 0
        pt_I[i, j] = 0

        vtdt_I[i, j] = [0,0]
        vt_I[i, j] = [0,0]

        utdt_I[i, j] = [0,0]
        ut_I[i, j] = [0,0]

        Delta_utdt_I[i, j] = [0,0]
        Delta_ut_I[i, j] = [0,0]

        atdt_I[i, j] = [0,0]
        at_I[i, j] = [0,0]
        ft_I[i, j] = [0,0]

        # volume0_0[i,j] = 0
        # volumet_0[i,j] = 0
        # cell[i,j] = 0

        # divvt_0_denominator[i, j] = 0
        # divvt_0_numerator[i, j] = 0
        # divvt_0[i, j] = 0

    if isPenaltyBC_elseBruteforceBC:
        penaltybc()

    for p in xtdt_p: 
        # cellBase = (xtdt_p[p] * inv_dx).cast(int)
        base = (xtdt_p[p] * inv_dx - shift).cast(int) # Define the bottom left corner of the surrounding 3x3 grid of neighboring nodes
        fx = (xtdt_p[p] * inv_dx - base.cast(ti.f64)) * dx # Define the vector from "base" to the current particle
        PartitionOfUnity[p] = ti.cast(0, ti.f64) # reset PoU, consistency, and gradient consistency for each particle
        Cons[p] = ti.cast(0, ti.f64)
        Cons_dx[p] = ti.cast(0, ti.f64)
        Cons_dy[p] = ti.cast(0, ti.f64)

        Psi_I, Psi_Icommax, Psi_Icommay = getRK(xtdt_p[p], base, a)

        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)): # for I \in 3 by 3 grid
            gridNode = [float(i+base[0]) * dx, float(j+base[1]) * dx] # Current grid node location
            B_I = ti.Vector([Psi_Icommax[i,j], Psi_Icommay[i,j]]) # Assemble a phi gradient vector
            PartitionOfUnity[p] += Psi_I[i,j] # POU
            Cons[p] += Psi_I[i,j] * gridNode[0] * gridNode[1] # Consistency
            Cons_dx[p] += B_I[0] * gridNode[0] * gridNode[1] # Gradient consistency
            Cons_dy[p] += B_I[1] * gridNode[0] * gridNode[1] # Gradient consistency

            offset = ti.Vector([i, j]) # Vector of grid node positions relative to "base" 
            dpos = offset.cast(ti.f64)*dx - fx # A vector from the current grid node to the current particle
            vt_I_APIC = Lt_p[p] @ dpos # define the contribution of the velocity gradient to the particle momentum 
            volumet_I[base + offset] += Psi_I[i,j] * volumet_p[p]
            mt_I[base + offset] += Psi_I[i,j] * mt_p[p] * ti.Matrix.identity(ti.f64, 2)
            vt_I[base + offset] += Psi_I[i,j] * mt_p[p] * (vtdt_p[p] + isAPIC_elsePIC * vt_I_APIC) # obtain $(mv)^t_I$
            at_I[base + offset] += Psi_I[i,j] * mt_p[p] * atdt_p[p]
            ut_I[base + offset] += Psi_I[i,j] * mt_p[p] * utdt_p[p]
            Delta_ut_I[base + offset] += Psi_I[i,j] * mt_p[p] * Delta_utdt_p[p]
            ft_I[base + offset] +=  volumet_p[p] * Psi_I[i,j] * fb[None] - volumet_p[p] * ( sigma[p] @ B_I )
            ptdt_I[base + offset] += Psi_I[i,j]*volumet_p[p]*ptdt_p[p] - dt*kappa*volumet_p[p]*Psi_I[i,j]*divvt_p[p]
            pt_I[base + offset] += Psi_I[i,j]*volumet_p[p]*ptdt_p[p]

        Cons[p] -= xtdt_p[p][0] * xtdt_p[p][1] # Consistency
        Cons_dx[p] -= float(1.0) * xtdt_p[p][1] # Gradient consistency
        Cons_dy[p] -= float(1.0) * xtdt_p[p][0] # Gradient consistency

    for i, j in mt_I:
        if volumet_I[i, j] != 0:
            ptdt_I[i,j] /= (volumet_I[i,j] + epsilon) # pressure-volume parameter to pressure
            pt_I[i,j] /= (volumet_I[i,j] + epsilon)
        if mt_I[i, j][0,0] != 0 and mt_I[i, j][1,1] != 0:# and volumet_I[i, j] != 0:
            atdt_I[i,j] = mt_I[i, j].inverse() @ ft_I[i,j]
            at_I[i,j] = mt_I[i, j].inverse() @ at_I[i,j]
            vt_I[i,j] = mt_I[i, j].inverse() @ vt_I[i,j]
            Delta_ut_I[i,j] = mt_I[i, j].inverse() @ Delta_ut_I[i,j]
            ut_I[i,j] = mt_I[i, j].inverse() @ ut_I[i,j]

            if isPenaltyBC_elseBruteforceBC == False:
                # if i < nodeNum: 
                    # at_I[i, j][0] = 0
                    # atdt_I[i, j][0] = 0
                    # if vt_I[i, j][0] < 0: vt_I[i, j][0] = 0
                # if i > num_g - nodeNum - 1: 
                    # at_I[i, j][0] = 0
                    # atdt_I[i, j][0] = 0
                    # if vt_I[i, j][0] > 0: vt_I[i, j][0] = 0
                # if j < nodeNum:
                    # at_I[i, j][1] = 0
                    # atdt_I[i, j][1] = 0
                    # if vt_I[i, j][1] < 0: vt_I[i, j][1] = 0
                # if j > num_g - nodeNum - 1: 
                    # at_I[i, j][1] = 0
                    # atdt_I[i, j][1] = 0
                    # if vt_I[i, j][1] > 0: vt_I[i, j][1] = 0

                if isCSL_elseMPM:
                    vtdt_I[i,j] = vt_I[i,j] + (1-gammaNewmark) * at_I[i,j] * dt + gammaNewmark * atdt_I[i,j] * dt
                    Delta_utdt_I[i,j] = vt_I[i,j]*dt + (0.5 - betaNewmark) * at_I[i,j] * dt**2 + betaNewmark * atdt_I[i,j] * dt**2 
                    utdt_I[i,j] = ut_I[i,j] + vt_I[i,j]*dt + (0.5 - betaNewmark) * at_I[i,j] * dt**2 + betaNewmark * atdt_I[i,j] * dt**2 
                    if i < nodeNum:
                        # atdt_I[i, j][0] = 0
                        if vtdt_I[i, j][0] < 0: vtdt_I[i, j][0] = 0
                        if Delta_utdt_I[i, j][0] < 0: Delta_utdt_I[i, j][0] = 0
                        if utdt_I[i, j][0] < 0: utdt_I[i, j][0] = ut_I[i,j][0]
                    if i > num_g - nodeNum - 1:
                        # atdt_I[i, j][0] = 0
                        if vtdt_I[i, j][0] > 0: vtdt_I[i, j][0] = 0
                        if Delta_utdt_I[i, j][0] > 0: Delta_utdt_I[i, j][0] = 0
                        if utdt_I[i, j][0] > 0: utdt_I[i, j][0] = ut_I[i,j][0]
                    if j < nodeNum:
                        # atdt_I[i, j][1] = 0
                        if vtdt_I[i, j][1] < 0: vtdt_I[i, j][1] = 0
                        if Delta_utdt_I[i, j][1] < 0: Delta_utdt_I[i, j][1] = 0
                        if utdt_I[i, j][1] < 0: utdt_I[i, j][1] = ut_I[i,j][1]
                    if j > num_g - nodeNum - 1:
                        # atdt_I[i, j][1] = 0
                        if vtdt_I[i, j][1] > 0: vtdt_I[i, j][1] = 0
                        if Delta_utdt_I[i, j][1] > 0: Delta_utdt_I[i, j][1] = 0
                        if utdt_I[i, j][1] > 0: utdt_I[i, j][1] = ut_I[i,j][1]
                        
                else:
                    vtdt_I[i,j] = vt_I[i,j] + atdt_I[i,j] * dt
                    if i < nodeNum and vtdt_I[i, j][0] < 0: 
                        vtdt_I[i, j][0] = 0
                    if i > num_g - nodeNum - 1 and vtdt_I[i, j][0] > 0: 
                        vtdt_I[i, j][0] = 0
                    if j < nodeNum and vtdt_I[i, j][1] < 0: 
                        vtdt_I[i, j][1] = 0
                    if j > num_g - nodeNum - 1 and vtdt_I[i, j][1] > 0: 
                        vtdt_I[i, j][1] = 0

    for p in xtdt_p: 
        base = (xtdt_p[p] * inv_dx - shift).cast(int) #每个 particle 所属的 3x3 support 的左下角点位置
        fx = ( xtdt_p[p] * inv_dx - base.cast(ti.f64) ) * dx # 向量，由 base 指向 particle

        vtdt_p_APIC = ti.Vector.zero(ti.f64, 2) # Initialize an APIC velocity vector
        vtdt_p_FLIP = ti.Vector.zero(ti.f64, 2) # Initialize a FLIP velocity vector
        new_L = ti.Matrix.zero(ti.f64, 2, 2) # Initialize a velocity gradient matrix
        new_divvt_p = ti.cast(0, ti.f64)
        new_ptdt_p = ti.cast(0, ti.f64)
        new_Delta_ptdt_p = ti.cast(0, ti.f64)
        new_atdt_p = ti.Vector.zero(float, 2) # Initialize an APIC velocity vector
        new_at_p = ti.Vector.zero(float, 2)
        new_utdt_p = ti.Vector.zero(float, 2)
        Delta_utdt_p_APIC = ti.Vector.zero(float, 2) 
        Delta_utdt_p_FLIP = ti.Vector.zero(float, 2) 
        Delta_Delta_utdt_p_FLIP = ti.Vector.zero(float, 2) 
        Psi_I, Psi_Icommax, Psi_Icommay = getRK(xtdt_p[p], base, a)

        for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)): 
            B_I = ti.Vector([Psi_Icommax[i,j], Psi_Icommay[i,j]]) # 2 by 1 vector, assemble a phi gradient vector
            offset = ti.Vector([i, j])
            new_L += vtdt_I[base + offset].outer_product(B_I) # define the velocity gradient
            new_divvt_p += vtdt_I[base + offset].dot(B_I)

            # if isCSLFLIPscheme2 == False:
            Delta_utdt_p_APIC += Psi_I[i,j] * (utdt_I[base + offset] - ut_I[base + offset])
            Delta_Delta_utdt_p_FLIP += Psi_I[i,j] * (Delta_utdt_I[base + offset] - Delta_ut_I[base + offset])
            
            vtdt_p_APIC += Psi_I[i,j] * (vtdt_I[base + offset]) # - 0.5 * at_I[base + offset] * dt**2 # APIC calculation of velocity
            vtdt_p_FLIP += Psi_I[i,j] * (vtdt_I[base + offset]  - vt_I[base + offset]) # FLIP calculation of velocity
            new_ptdt_p += Psi_I[i,j] * ptdt_I[base + offset]
            new_Delta_ptdt_p += Psi_I[i,j] * (ptdt_I[base + offset] - pt_I[base + offset])
            new_atdt_p += Psi_I[i,j] * atdt_I[base + offset]
            new_at_p += Psi_I[i,j] * at_I[base + offset]
            new_utdt_p += Psi_I[i,j] * utdt_I[base + offset]

        

        if isCSL_elseMPM:
            # if isCSLFLIPscheme2: at_p[p] = new_at_p # atdt_p[p]
            
            atdt_p[p] = new_atdt_p
            utdt_p[p] = new_utdt_p
            Delta_utdt_p_FLIP = Delta_utdt_p[p] + Delta_Delta_utdt_p_FLIP
            Delta_utdt_p[p] =  float(eta_u)*Delta_utdt_p_FLIP + float(1-eta_u)*Delta_utdt_p_APIC
            # if isCSLFLIPscheme2:
            #     Delta_utdt_p = vtdt_p[p]*dt + (0.5-betaNewmark)*at_p[p]*dt**2 + betaNewmark*atdt_p[p]*dt**2
            #     # Delta_utdt_p = vtdt_p[p]*dt + betaNewmark*atdt_p[p]*dt**2
            #     xtdt_p[p] += Delta_utdt_p
            #     vtdt_p[p] += dt * atdt_p[p]
            #     # xtdt_p[p] += dt * vtdt_p[p]
            # else:
            xtdt_p[p] += Delta_utdt_p[p] #　Delta_utdt_p[p]
            # vtdt_p[p] += dt * atdt_p[p]
            vtdt_p[p] += vtdt_p_FLIP # dt * atdt_p[p] # vtdt_p_FLIP
            vtdt_p[p] = float(eta_v)*vtdt_p[p] + float(1-eta_v)*vtdt_p_APIC
            # vtdt_p[p] = vtdt_p_APIC
            # xtdt_p[p] += dt * vtdt_p[p]
        else:
            vtdt_p_FLIP += vtdt_p[p]
            vtdt_p[p] = float(eta_v)*vtdt_p_FLIP + float(1-eta_v)*vtdt_p_APIC # Define the particle velocity (FLIP blend)
            xtdt_p[p] += dt * vtdt_p[p]

        Lt_p[p] = new_L
        Ft_p[p] = (ti.Matrix.identity(ti.f64, 2) + dt * Lt_p[p]) @ Ft_p[p] # Deformation gradient update
        U_F, sig_F, V_F = ti.svd(Ft_p[p]) # Singular value decomposition
        J = ti.cast(1.0, ti.f64)
        for d in ti.static(range(dim)):
            J *= sig_F[d, d]
        detF[p] = J
        volumet_p[p] = volume0_p * detF[p] # Update particle volumes using F
        rho_p[p] = rho / (detF[p] + epsilon)
        mt_p[p] = volumet_p[p] * rho_p[p]
        
        if isInterTimeStepDivv:
            divvt_p[p] = (1 - deltaSL) * divvt_p[p] + deltaSL * new_divvt_p
        else:
            divvt_p[p] = new_divvt_p

        pt_p[p] = ptdt_p[p]
        if isMixedFormulation_elsePointwise:
            ptdt_p[p] = float(eta_p)*(ptdt_p[p] + new_Delta_ptdt_p) + float(1-eta_p)*new_ptdt_p
        else:
            ptdt_p[p] -= dt*kappa*divvt_p[p]
            # ptdt_p[p] = n_const*1540**2/rho*((rho_p[p]/rho)**n_const - 1)

    for p in xtdt_p: 
        vNRLAV = ti.cast(0, ti.f64)
        if ifAV == 1:
            if divvt_p[p] <0:
                vNRLAV = -rho*c_L*dx*c_artificial*divvt_p[p] + rho*c_Q*dx**2*divvt_p[p]**2
            else:
                vNRLAV = 0
        sigma[p] = - (ptdt_p[p] + ifAV * vNRLAV) * ti.Matrix.identity(ti.f64, 2) + visc * (Lt_p[p] + Lt_p[p].transpose())

    # if isDivvBar:
    #     for p in xtdt_p:
    #         # cellBase = (xtdt_p[p] * inv_dx).cast(int)
    #         base = (xtdt_p[p] * inv_dx - shift).cast(int)
    #         Psi_I, Psi_Icommax, Psi_Icommay = getRK(xtdt_p[p], base, a)
    #         for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
    #             offset = ti.Vector([i, j])
    #             divvt_0_denominator[base + offset] += volume0_p * Psi_I[i,j]
    #             divvt_0_numerator[base + offset] += divvt_p[p] * volume0_p * Psi_I[i,j]
    #     for i, j in divvt_0:
    #         divvt_0[i,j] = divvt_0_numerator[i,j] / (divvt_0_denominator[i,j] + epsilon)
    #     for p in xtdt_p:
    #         # cellBase = (xtdt_p[p] * inv_dx).cast(int)
    #         base = (xtdt_p[p] * inv_dx - shift).cast(int)
    #         Psi_I, Psi_Icommax, Psi_Icommay = getRK(xtdt_p[p], base, a)
    #         for i, j in ti.static(ti.ndrange(nodeNum, nodeNum)):
    #             offset = ti.Vector([i, j])
    #             divvt_p[p] += Psi_I[i,j] * divvt_0[base + offset]

x_pos1 = ti.field(dtype=ti.f64, shape=(np_x, np_y))
y_pos1 = ti.field(dtype=ti.f64, shape=(np_x, np_y))

@ti.kernel
def initialize_Cubes():
    for a, b in x_pos1:
        x_pos1[a,b] = (a/(np_x - 1))*W_fluid + 2*dx + (len_domain-W_fluid)
        y_pos1[a,b] = (b/(np_y - 1))*H_fluid + 2*dx
        
    for i in range(num_p):
        col = i - np_x * ( i // np_x )
        row = i // np_x
        xtdt_p[i] = [x_pos1[col,row], y_pos1[col,row]] # xtdt_p[i] = [ ti.random() * Liquid_Width + 2 * dx, ti.random() * Liquid_Height + 2 * dx]       # Random distribution
        vtdt_p[i] = [0, 0] # vtdt_p[i] = ti.Matrix([[ti.cos(ti.math.pi), -ti.sin(ti.math.pi)], [ti.sin(ti.math.pi), ti.cos(ti.math.pi)]]) @ xt_p[i] / dt
        mt_p[i] = volume0_p * rho
        material[i] = 0
        volumet_p[i] = volume0_p
        Ft_p[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    
    for i in x_L_left:
        x_L_left[i] = [2 * dx, (2.5 + i) * dx]
        x_L_right[i] = [len_domain + 2 * dx, (2.5 + i) * dx]
        x_L_bot[i] = [(2.5 + i) * dx, 2 * dx]
        x_L_top[i] = [(2.5 + i) * dx, len_domain + 2 * dx]
        
# ------------------------------------
initialize_Cubes()

# GUI setup
gui = ti.GUI("Window Title", res=512, show_gui=False, background_color=0xFFFFFF)

# Ensure output directories exist
os.makedirs(filepath, exist_ok=True)
os.makedirs(vtkpath, exist_ok=True)

# Define the progress bar function
def progressBar(progress, total, color=colorama.Fore.YELLOW):
    percentage = 100 * (progress / float(total))
    bar = '█' * int(percentage) + '-' * (100 - int(percentage))
    print(color + f"\r|{bar}| {percentage:.2f}%" + " | Current Time: " + str(progress))
    sys.stdout.write("\033[F")  # Move back to previous line
    sys.stdout.write("\033[K")  # Clear line
    if progress >= total:
        print(colorama.Fore.GREEN + f"\r|{bar}| {percentage:.2f}%" + " | It's done! | Current time: " + str(progress), end="\r")
        print(colorama.Fore.RED)

# Initialize simulation variables
# pressureCheck = []
count = 0
# pressureCount = 0

T_values = []
L_values = []
H_values = []

# Run simulation loop
while timeTotal < simTime:
    # pressureCount = 0
    num_substeps = int(1e-2 // dt)
    
    # Combine data operations where possible
    xtdt_p_np = xtdt_p.to_numpy()
    vtdt_p_np = vtdt_p.to_numpy()
    sigma_np = sigma.to_numpy()
    PartitionOfUnity_np = PartitionOfUnity.to_numpy()
    Cons_np = Cons.to_numpy()
    Cons_dx_np = Cons_dx.to_numpy()
    Cons_dy_np = Cons_dy.to_numpy()
    atdt_p_np = atdt_p.to_numpy()
    detF_np = detF.to_numpy()
    material_np = material.to_numpy()
    
    for s in range(num_substeps):  # Sub-steps
        substep()
        count += 1
        timeTotal += dt

    l_T = np.max(xtdt_p_np[:, 0]) - np.min(xtdt_p_np[:, 0])
    h_T = np.max(xtdt_p_np[:, 1]) - np.min(xtdt_p_np[:, 1])
    
    T = timeTotal * np.sqrt(H_fluid * (-a_g) / (W_fluid**2))
    L = l_T / W_fluid
    H = h_T / H_fluid
    
    T_values.append(T)
    L_values.append(L)
    H_values.append(H)

    # Collect data
    # ptdt_array = ptdt_I.to_numpy()
    # mid_x = (ptdt_array.shape[0] - 1) // 2
    # mid_y = (ptdt_array.shape[1] - 1) // 2
    # pressureCheck.append(ptdt_array[mid_x, mid_y])
    print('Current Time: ', timeTotal)
    
    # Prepare data for VTK and visualization
    xCoordinate = np.ascontiguousarray(xtdt_p_np[:, 0])
    yCoordinate = np.ascontiguousarray(xtdt_p_np[:, 1])
    zCoordinate = np.ascontiguousarray(np.zeros(num_p))
    v_x = np.ascontiguousarray(vtdt_p_np[:, 0])
    v_y = np.ascontiguousarray(vtdt_p_np[:, 1])
    v_z = np.ascontiguousarray(np.zeros(num_p))
    pressure = np.ascontiguousarray(-(sigma_np[:, 0, 0] + sigma_np[:, 1, 1]) / 3)

    sigma_11 = np.ascontiguousarray(sigma_np[:, 0, 0])
    sigma_22 = np.ascontiguousarray(sigma_np[:, 1, 1])
    sigma_12 = np.ascontiguousarray(sigma_np[:, 0, 1])

    PoU = np.ascontiguousarray(PartitionOfUnity_np[:] - 1.0)
    Consistency = np.ascontiguousarray(Cons_np[:])
    Consistency_Gradx = np.ascontiguousarray(Cons_dx_np[:])
    Consistency_Grady = np.ascontiguousarray(Cons_dy_np[:])
    accDisplay_y = np.ascontiguousarray(atdt_p_np[:, 1])

    # Calculate vp_mag_step
    vp_mag_step = np.ascontiguousarray(np.sqrt(v_x**2 + v_y**2 + v_z**2))

    # Save data to VTK less frequently
    if count % (num_substeps) == 0:  # Save every 100th frame (adjust as needed)
        pointsToVTK(
            f'./{vtkpath}/points{gui.frame:06d}',
            xCoordinate, yCoordinate, zCoordinate,
            data={
                "ID": np.ascontiguousarray(material_np),
                "simgaxx": sigma_11,
                "simgayy": sigma_22,
                "simgaxy": sigma_12,
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
                "Pressure": pressure,
                "Partition of Unity": PoU,
                "Consistency": Consistency,
                "Gradx Consistency": Consistency_Gradx,
                "Grady Consistency": Consistency_Grady,
                "Deformation": np.ascontiguousarray(detF_np),
                "Velocity Mag": vp_mag_step,
                "Acceleration Y": accDisplay_y
            }
        )

    # Render GUI less frequently
    # if count % (num_substeps) == 0:  # Render every 100th frame (adjust as needed)
    #     colors = np.array([0x000000] * len(material_np), dtype=np.uint32)  # Set all particles to black
    #     gui.circles(xtdt_p_np, radius=0.8, color=colors)
    #     gui.show(filepath + f'/{gui.frame:06d}.png')
    if count % (num_substeps) == 0:  # Render every 100th frame (adjust as needed)
        colors = np.array([0x000000] * len(material_np), dtype=np.uint32)
        
        # Scale coordinates to fit the 1 by 1 window
        scaled_xtdt_p_np = xtdt_p_np / 0.5  # Scaling the coordinates
        
        gui.circles(scaled_xtdt_p_np, radius=0.8, color=colors)
        gui.show(filepath + f'/{gui.frame:06d}.png')

# Save runtime
time1 = time.time()
print('Run Time:', time1 - time0)

# Save T, L(T), and H(T) to CSV using pandas
data = {
    "T": T_values,
    "L(T)": L_values,
    "H(T)": H_values
}
df = pd.DataFrame(data)
csv_name = f"water_column_{vtkpath}_data.csv"
df.to_csv(csv_name, index=False)