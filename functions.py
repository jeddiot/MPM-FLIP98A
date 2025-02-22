import taichi as ti
import sys
import colorama
import math
import numpy as np
from pyevtk.hl import pointsToVTK

from config import NumericalSettings, PhysicalQuantities, GravityField
from fields import ParticleFields, GridFields, StabilizationFields, ProjectionFields, PenaltyMethodFields

physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
gravitational = GravityField(numerical, physical)
particle = ParticleFields(numerical.numParticles)
grid = GridFields(numerical.numGrids)
stability = StabilizationFields(numerical.numGrids)
projection = ProjectionFields(numerical.numGrids)
penalty = PenaltyMethodFields(numerical.numCells)


@ti.func
def reproducingKernelFunction(position, base, kernelSupportSize, gridSpacing, nodeCount, dimension, switch_getRK_Bspline):

    # nodeCount = int(nodeCount)
    # dimension = int(dimension)

    # Initialize the matrix of phi values for each surrounding grid node at the current particle location
    phiMat = ti.Matrix.zero(ti.f64, 3, 3)
    # Initialize the matrix of dphi/dx values for each surrounding grid node at the current particle location
    dphiXMat = ti.Matrix.zero(ti.f64, 3, 3)
    # Initialize the matrix of dphi/dy values for each surrounding grid node at the current particle location
    dphiYMat = ti.Matrix.zero(ti.f64, 3, 3)
    # Initialize the moment matrix
    momentMatrix = ti.Matrix.zero(ti.f64, 3, 3)

    kerneFunctionWeight1D = ti.Matrix.zero(ti.f64, 3, 2)
    kerneFunctionWeight2D = ti.Matrix.zero(ti.f64, 3, 3)
    for i, d in ti.static(ti.ndrange(3, 2)):
        # Current grid node location
        gridNode = float(i + base[d]) * gridSpacing
        if gridNode >= 0:
            # Normalized distance from particle to grid node
            z = abs(position[d] - float(gridNode)) / kernelSupportSize
            if switch_getRK_Bspline:
                if 0 <= z and z < 0.5:                      # Cubic b-spline kernel functions
                    kerneFunctionWeight1D[i, d] = 2/3 - 4*z**2 + 4*z**3
                elif 1/2 <= z and z < 1:
                    kerneFunctionWeight1D[i, d] = 4/3 - 4*z + 4*z**2 - (4/3)*z**3
            else:
                if 0 <= z and z < 1:
                    kerneFunctionWeight1D[i, d] = z - 1
                elif 1 <= z:
                    kerneFunctionWeight1D[i, d] = 0
    for i, j in ti.static(ti.ndrange(3, 3)):
        gridNode = [float(i+base[0]) * gridSpacing, float(j+base[1])
                    * gridSpacing]  # Current grid node location
        if gridNode[0] >= 0 and gridNode[1] >= 0:
            kerneFunctionWeight2D[i, j] = kerneFunctionWeight1D[i, 0] * kerneFunctionWeight1D[j, 1]
            # Define P(xi - xp)
            Pxi = ti.Vector([1.0, position[0] - gridNode[0], position[1] - gridNode[1]])
            if kerneFunctionWeight2D[i, j] != 0:
                # momentMatrix += weight[i, j] * Pxi @ Pxi.transpose()
                momentMatrix += kerneFunctionWeight2D[i, j] * Pxi.outer_product(Pxi)

    momentMatrix_inv = momentMatrix.inverse()
    # Loop over neighboring grid nodes
    for i, j in ti.static(ti.ndrange(3, 3)):
        gridNode = [float(i+base[0]) * gridSpacing, float(j+base[1])
                    * gridSpacing]  # Current grid node location
        if kerneFunctionWeight2D[i, j] != 0:
            if gridNode[0] >= 0 and gridNode[1] >= 0:
                # Define P(xi - xp)
                Pxi = ti.Vector([1.0, position[0] - gridNode[0], position[1] - gridNode[1]])
                Pxp = ti.Vector([1.0, position[0] - position[0], position[1] - position[1]])
                dPxpX = ti.Vector([0.0, -1.0, 0.0])
                dPxpY = ti.Vector([0.0, 0.0, -1.0])

                phi = kerneFunctionWeight2D[i, j] * (Pxp.dot(momentMatrix_inv @ Pxi))
                dphi_x1 = kerneFunctionWeight2D[i, j] * (dPxpX.dot(momentMatrix_inv @ Pxi))
                dphi_x2 = kerneFunctionWeight2D[i, j] * (dPxpY.dot(momentMatrix_inv @ Pxi))

                phiMat[i, j] = phi
                dphiXMat[i, j] = dphi_x1
                dphiYMat[i, j] = dphi_x2

    return phiMat, dphiXMat, dphiYMat


@ti.func
def penaltyBoundary(boundary, gridSpacing, gridNodeShift, kernelSupportSize, numCells, nodeCount, dimension, switch_getRK_Bspline):
    for i in boundary:
        base = (boundary[i] / gridSpacing - gridNodeShift).cast(int)
        shapeFunction_grid, _, _ = reproducingKernelFunction(
            boundary[i], base, kernelSupportSize, gridSpacing, nodeCount, dimension, switch_getRK_Bspline)
        s1 = 1
        s2 = 0
        S = ti.Matrix([[s1, 0], [0, s2]])
        for i, j in ti.static(ti.ndrange(3, 3)): #nodeCount
            offset = ti.Vector([i, j])

            if boundary == penalty.left_boundary:
                if float(base[0] + offset[0]) >= 2:
                    grid.mass_grid[base + offset] += gridSpacing * \
                        numerical.penalty * shapeFunction_grid[i, j] * S
                if float(base[0] + offset[0]) < 2:
                    grid.mass_grid[base +
                                       offset] += gridSpacing * numerical.penalty * S

            if boundary == penalty.right_boundary:
                if float(base[0] + offset[0]) <= numCells - 2:
                    grid.mass_grid[base + offset] += gridSpacing * \
                        numerical.penalty * shapeFunction_grid[i, j] * S
                if float(base[0] + offset[0]) > numCells - 2:
                    grid.mass_grid[base +
                                       offset] += gridSpacing * numerical.penalty * S

            if boundary == penalty.bottom_boundary:
                if float(base[1] + offset[1]) >= 2:
                    grid.mass_grid[base + offset] += gridSpacing * \
                        numerical.penalty * shapeFunction_grid[i, j] * S
                if float(base[1] + offset[1]) < 2:
                    grid.mass_grid[base +
                                       offset] += gridSpacing * numerical.penalty * S

            if boundary == penalty.top_boundary:
                if float(base[1] + offset[1]) <= numCells - 2:
                    grid.mass_grid[base + offset] += gridSpacing * \
                        numerical.penalty * shapeFunction_grid[i, j] * S
                if float(base[1] + offset[1]) > numCells - 2:
                    grid.mass_grid[base +
                                       offset] += gridSpacing * numerical.penalty * S


def format_exp(x, n, d=6):
    significand = x / 10 ** n
    exp_sign = '+' if n >= 0 else ''
    # exp_sign = '+' if n >= 0 else '-'
    return f'{significand:.{d}f}e{exp_sign}{n:02d}'


def createFilePaths(numerical: NumericalSettings):
    filepath = "mov"
    vtkpath = "vtk"

    if numerical.pressureMixingRatio == 1:
        filepath = filepath + "_mixed"
        vtkpath = vtkpath + "_mixed"
    elif numerical.pressureMixingRatio == 0:
        filepath = filepath + "_pointwise"
        vtkpath = vtkpath + "_pointwise"

    filepath = filepath + "_dt" + format_exp(numerical.timeStep,
                                             math.floor(math.log10(numerical.timeStep)), 0)
    vtkpath = vtkpath + "_dt" + \
        format_exp(numerical.timeStep, math.floor(
            math.log10(numerical.timeStep)), 0)

    if numerical.switch_penaltyEBC:
        filepath = filepath + "_betaNor" + format_exp(numerical.penalty,
                                                      math.floor(math.log10(numerical.penalty)), 0)
        vtkpath = vtkpath + "_betaNor" + format_exp(numerical.penalty,
                                                    math.floor(math.log10(numerical.penalty)), 0)

    return filepath, vtkpath


def progressBar(progress, total, color=colorama.Fore.YELLOW):
    percentage = 100 * (progress/float(total))
    bar = '█' * int(percentage) + '-'*(100 - int(percentage))
    print(color + f"\r|{bar}| {percentage:.2f}%" +
          " | Current Time: " + str(progress))  # , end = "\r")
    sys.stdout.write("\033[F")  # back to previous line
    sys.stdout.write("\033[K")  # clear line
    if progress >= total:
        print(colorama.Fore.GREEN + f"\r|{bar}| {percentage:.2f}%" +
              " | It's done ! | Current time: " + str(progress), end="\r")
        print(colorama.Fore.RED)


positionX = ti.field(dtype=ti.f64, shape=(
    numerical.numParticlesX, numerical.numParticlesY))
positionY = ti.field(dtype=ti.f64, shape=(
    numerical.numParticlesX, numerical.numParticlesY))


@ti.kernel
def initialize_Cubes():
    for a, b in positionX:
        positionX[a, b] = (a / (numerical.numParticlesX - 1)) * \
            numerical.fluidWidth + 2 * numerical.gridSpacing
        positionY[a, b] = (b / (numerical.numParticlesY - 1)) * \
            numerical.fluidHeight + 2 * numerical.gridSpacing

    for i in range(numerical.numParticles):
        col = i - numerical.numParticlesX * (i // numerical.numParticlesY)
        row = i // numerical.numParticlesX
        particle.position[i] = [positionX[col, row], positionY[col, row]]
        particle.position[i] = [ti.random() * numerical.fluidWidth + 2 * numerical.gridSpacing,
                                ti.random() * numerical.fluidHeight + 2 * numerical.gridSpacing]       # Random distribution
        particle.velocity[i] = [0, 0]
        particle.mass[i] = numerical.initialParticleVolume * \
            physical.particleDensity
        particle.material_id[i] = 0
        particle.volume[i] = numerical.initialParticleVolume
        particle.deformation_gradient[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

    for i in penalty.left_boundary:
        penalty.left_boundary[i] = [
            2 * numerical.gridSpacing, (2.5 + i) * numerical.gridSpacing]
        penalty.right_boundary[i] = [numerical.domainLength + 2 *
                                     numerical.gridSpacing, (2.5 + i) * numerical.gridSpacing]
        penalty.bottom_boundary[i] = [
            (2.5 + i) * numerical.gridSpacing, 2 * numerical.gridSpacing]
        penalty.top_boundary[i] = [
            (2.5 + i) * numerical.gridSpacing, numerical.domainLength + 2 * numerical.gridSpacing]
        
def post_process(numerical, particle, gui, vtkpath, filepath):
    xCoordinate = np.zeros(particle.numParticles)
    yCoordinate = np.zeros(particle.numParticles)
    zCoordinate = np.zeros(particle.numParticles)
    v_x = np.zeros(particle.numParticles)
    v_y = np.zeros(particle.numParticles)
    v_z = np.zeros(particle.numParticles)
    materialID = np.zeros(particle.numParticles)
    deformation_step = np.zeros(particle.numParticles)
    PoU = np.zeros(particle.numParticles)
    Consistency = np.zeros(particle.numParticles)
    Consistency_Gradx = np.zeros(particle.numParticles)
    Consistency_Grady = np.zeros(particle.numParticles)
    vp_mag_step = np.zeros(particle.numParticles)
    pressure = np.zeros(particle.numParticles)
    
    sigma_11 = np.zeros(particle.numParticles)
    sigma_22 = np.zeros(particle.numParticles)
    sigma_12 = np.zeros(particle.numParticles)
    
    xCoordinate[:] = particle.position.to_numpy()[:, 0]
    yCoordinate[:] = particle.position.to_numpy()[:, 1]
    zCoordinate[:] = 0 * particle.position.to_numpy()[:, 1]
    deformation_step[:] = particle.determinant_of_deformation_gradient.to_numpy()[:]
    v_x[:] = particle.velocity.to_numpy()[:, 0]
    v_y[:] = particle.velocity.to_numpy()[:, 1]
    v_z[:] = 0 * particle.velocity.to_numpy()[:, 1]
    pressure[:] = - (particle.stress.to_numpy()[:, 0, 0] +
                     particle.stress.to_numpy()[:, 1, 1]) / 3
    
    sigma_11[:] = particle.stress.to_numpy()[:, 0, 0]
    sigma_22[:] = particle.stress.to_numpy()[:, 1, 1]
    sigma_12[:] = particle.stress.to_numpy()[:, 0, 1]
    
    PoU[:] = particle.partitionofUnity.to_numpy()[:] - float(1.0)
    Consistency[:] = particle.consistency.to_numpy()[:]
    Consistency_Gradx[:] = particle.consistency_dx.to_numpy()[:]
    Consistency_Grady[:] = particle.consistency_dy.to_numpy()[:]
    
    pointsToVTK(f'./{vtkpath}/points{gui.frame:06d}', xCoordinate, yCoordinate, zCoordinate,
                data={"ID": particle.material_id.to_numpy(), "simgaxx": sigma_11, "simgayy": sigma_22, "simgaxy": sigma_12,
                      "v_x": v_x, "v_y": v_y, "v_z": v_z, "Pressure": pressure, "Partition of Unity": PoU, "Consistency": Consistency,
                      "Gradx Consistency": Consistency_Gradx, "Grady Consistency": Consistency_Grady, "Deformation": deformation_step,
                      "Velocity Mag": vp_mag_step},
                fieldData={"Velocity": np.concatenate((v_x, v_y, v_z), axis=0)})
    
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    gui.circles(particle.position.to_numpy(), radius=0.8,
                color=colors[particle.material_id.to_numpy()])
    gui.show(f'{filepath}/{gui.frame:06d}.png')


@ti.kernel
def substep():
    # reset after every timestep
    for i, j in grid.mass_grid:
        grid.velocity_grid[i, j] = [0, 0]
        grid.velocity_grid_initial[i, j] = [0, 0]
        grid.mass_grid[i, j] = [[0, 0], [0, 0]]
        grid.volume_grid[i, j] = 0
        grid.pressure_grid[i, j] = 0

    if numerical.switch_penaltyEBC:
        penaltyBoundary(penalty.left_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.nodeCount, numerical.dimension, numerical.switch_getRK_Bspline)
        penaltyBoundary(penalty.right_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.nodeCount, numerical.dimension, numerical.switch_getRK_Bspline)
        penaltyBoundary(penalty.bottom_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.nodeCount, numerical.dimension, numerical.switch_getRK_Bspline)
        penaltyBoundary(penalty.top_boundary, numerical.gridSpacing, numerical.gridNodeShift, numerical.kernelSupportSize,
                        numerical.numCells, numerical.nodeCount, numerical.dimension, numerical.switch_getRK_Bspline)

    for p in particle.position:
        cellBase = (particle.position[p] / numerical.gridSpacing).cast(int)
        # Define the bottom left corner of the surrounding 3x3 grid of neighboring nodes
        base = (particle.position[p] / numerical.gridSpacing -
                numerical.gridNodeShift).cast(int)
        vector_base2CurrentParticle = (particle.position[p] / numerical.gridSpacing - base.cast(ti.f64)) * \
            numerical.gridSpacing

        particle.partitionofUnity[p] = ti.cast(0, ti.f64)
        particle.consistency[p] = ti.cast(0, ti.f64)
        particle.consistency_dx[p] = ti.cast(0, ti.f64)
        particle.consistency_dy[p] = ti.cast(0, ti.f64)

        shapeFunction_grid, shapeFunction_grid_dx, shapeFunction_grid_dy = reproducingKernelFunction(
            particle.position[p], base, numerical.kernelSupportSize, numerical.gridSpacing, numerical.nodeCount,
            numerical.dimension, numerical.switch_getRK_Bspline)

        for i, j in ti.static(ti.ndrange(numerical.nodeCount, numerical.nodeCount)):
            currentGridNodeLocation = [float(i + base[0]) * numerical.gridSpacing, float(j + base[1])
                                       * numerical.gridSpacing]  # Current grid node location
            shapeFunction_grid_gradient = ti.Vector(
                # Assemble a phi gradient vector
                [shapeFunction_grid_dx[i, j], shapeFunction_grid_dy[i, j]])
            particle.partitionofUnity[p] += shapeFunction_grid[i, j]
            particle.consistency[p] += shapeFunction_grid[i, j] * \
                currentGridNodeLocation[0] * currentGridNodeLocation[1]
            particle.consistency_dx[p] += shapeFunction_grid_gradient[0] * \
                currentGridNodeLocation[0] * currentGridNodeLocation[1]
            particle.consistency_dy[p] += shapeFunction_grid_gradient[1] * \
                currentGridNodeLocation[0] * currentGridNodeLocation[1]

            # Vector of grid node positions relative to "base"
            offset = ti.Vector([i, j])
            # A vector from the current grid node to the current particle
            dpos = offset.cast(ti.f64) * numerical.gridSpacing - \
                vector_base2CurrentParticle

            # define the contribution of the velocity gradient to the particle momentum
            APIC_velocity_grid = particle.velocity_gradient[p] @ dpos

            grid.volume_grid[base + offset] += shapeFunction_grid[i,j] * particle.volume[p]
            grid.mass_grid[base + offset] += shapeFunction_grid[i,j] * particle.mass[p] * ti.Matrix.identity(ti.f64, 2)

            if numerical.switch_vt_I_APIC:
                grid.velocity_grid_initial[base + offset] += shapeFunction_grid[i, j] * particle.mass[p] * \
                    (particle.velocity[p] +
                     numerical.switch_vt_I_APIC * APIC_velocity_grid)
                grid.velocity_grid[base + offset] += shapeFunction_grid[i, j] * particle.mass[p] * \
                    (particle.velocity[p] +
                     numerical.switch_vt_I_APIC * APIC_velocity_grid)
            else:
                grid.velocity_grid_initial[base + offset] += shapeFunction_grid[i, j] * particle.mass[p] * \
                    (particle.velocity[p])
                grid.velocity_grid[base + offset] += shapeFunction_grid[i, j] * particle.mass[p] * \
                    (particle.velocity[p])

            grid.velocity_grid[base + offset] += numerical.timeStep * particle.volume[p] * \
                shapeFunction_grid[i, j] * gravitational.gravityField[0] - numerical.timeStep * \
                particle.volume[p] * (particle.stress[p]
                                      @ shapeFunction_grid_gradient)

            grid.pressure_grid[base + offset] += shapeFunction_grid[i, j] * particle.volume[p] * \
                particle.pressure[p] - numerical.timeStep * physical.bulkModulus * \
                particle.volume[p] * shapeFunction_grid[i,
                                                        j] * particle.divergenceofVelocity[p]

        particle.consistency[p] -= particle.position[p][0] * \
            particle.position[p][1]
        particle.consistency_dx[p] -= float(1.0) * particle.position[p][1]
        particle.consistency_dy[p] -= float(1.0) * particle.position[p][0]

    for i, j in grid.mass_grid:
        if grid.volume_grid[i, j] != 0:
            grid.pressure_grid[i, j] /= grid.volume_grid[i, j]
        if grid.mass_grid[i, j][0, 0] != 0 and grid.mass_grid[i, j][1, 1] != 0:
            grid.velocity_grid[i, j] = grid.mass_grid[i,
                                                      j].inverse() @ grid.velocity_grid[i, j]
            if numerical.switch_penaltyEBC:
                pass
            else:
                if i < numerical.nodeCount and grid.velocity_grid[i, j][0] < 0:
                    grid.velocity_grid[i, j][0] = 0
                if i > numerical.numGrids - numerical.nodeCount - 1 and grid.velocity_grid[i, j][0] > 0:
                    grid.velocity_grid[i, j][0] = 0
                if j < numerical.nodeCount and grid.velocity_grid[i, j][1] < 0:
                    grid.velocity_grid[i, j][1] = 0
                if j > numerical.numGrids - numerical.nodeCount - 1 and grid.velocity_grid[i, j][1] > 0:
                    grid.velocity_grid[i, j][1] = 0

    for p in particle.position:
        # the bottom left vertex of a 3x3 support for each particle
        base = (particle.position[p] / numerical.gridSpacing -
                numerical.gridNodeShift).cast(int)
        vector_base2CurrentParticle = (
            particle.position[p] / numerical.gridSpacing - base.cast(ti.f64)) * numerical.gridSpacing

        velocity_APIC = ti.Vector.zero(ti.f64, 2)
        velocity_FLIP = ti.Vector.zero(ti.f64, 2)
        new_velocity_gradient = ti.Matrix.zero(ti.f64, 2, 2)
        new_divergenceofVelocity = ti.cast(0, ti.f64)
        new_pressure = ti.cast(0, ti.f64)

        shapeFunction_grid, shapeFunction_grid_dx, shapeFunction_grid_dy = reproducingKernelFunction(
            particle.position[p], base, numerical.kernelSupportSize, numerical.gridSpacing, numerical.nodeCount,
            numerical.dimension, numerical.switch_getRK_Bspline)

        for i, j in ti.static(ti.ndrange(numerical.nodeCount, numerical.nodeCount)):
            shapeFunction_grid_gradient = ti.Vector(
                [shapeFunction_grid_dx[i, j], shapeFunction_grid_dy[i, j]])
            offset = ti.Vector([i, j])
            new_velocity_gradient += grid.velocity_grid[base + offset].outer_product(
                shapeFunction_grid_gradient)  # define the velocity gradient
            new_divergenceofVelocity += grid.velocity_grid[base + offset].dot(
                shapeFunction_grid_gradient)

            velocity_APIC += shapeFunction_grid[i, j] * grid.velocity_grid[base + offset]
            velocity_FLIP += shapeFunction_grid[i, j] * (grid.velocity_grid[base + offset] - (grid.mass_grid[base + offset].inverse() @
                                                                                              grid.velocity_grid_initial[base + offset]))
            new_pressure += shapeFunction_grid[i, j] * grid.pressure_grid[base + offset]

        velocity_FLIP += particle.velocity[p]
        particle.velocity[p] = numerical.flipBlendParameter * velocity_FLIP + (
            # Define the particle velocity (FLIP blend)
            1 - numerical.flipBlendParameter) * velocity_APIC
        particle.position[p] += numerical.timeStep * \
            particle.velocity[p]  # Advect particles NFLIP

        particle.velocity_gradient[p] = new_velocity_gradient
        particle.deformation_gradient[p] = (ti.Matrix.identity(ti.f64, 2) + numerical.timeStep * particle.velocity_gradient[p]) \
            @ particle.deformation_gradient[p]

        # # Singular value decomposition
        # _, sig_F, _ = ti.svd(particle.deformation_gradient[p])
        # jacobian = ti.cast(1.0, ti.f64)
        # for d in ti.static(range(numerical.dimension)):
        # jacobian *= sig_F[d, d]

        jacobian = particle.deformation_gradient[p][0,0] * particle.deformation_gradient[p][1, 1] - \
            particle.deformation_gradient[p][1, 0] * particle.deformation_gradient[p][0, 1]
        particle.determinant_of_deformation_gradient[p] = jacobian
        particle.volume[p] = numerical.initialParticleVolume * \
            particle.determinant_of_deformation_gradient[p]  # Update particle volumes using F
        particle.particleDensity[p] = physical.particleDensity / \
            particle.determinant_of_deformation_gradient[p]
        particle.mass[p] = particle.volume[p] * particle.particleDensity[p]

        particle.divergenceofVelocity[p] = new_divergenceofVelocity
        particle.pressure[p] -= numerical.timeStep * \
            physical.bulkModulus * particle.divergenceofVelocity[p]
        particle.pressure[p] = numerical.pressureMixingRatio * new_pressure + \
            (1 - numerical.pressureMixingRatio) * particle.pressure[p]

        strainRate = 0.5 * (
            particle.velocity_gradient[p] +
            ti.Matrix([[particle.velocity_gradient[p][0, 0], particle.velocity_gradient[p][1, 0]],
                       [particle.velocity_gradient[p][0, 1], particle.velocity_gradient[p][1, 1]]])
        )

        particle.stress[p] = - particle.pressure[p] * ti.Matrix.identity(ti.f64, 2) + 2 * physical.dynamicViscosity * strainRate