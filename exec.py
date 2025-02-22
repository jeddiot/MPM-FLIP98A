import numpy as np
import os as os
import time
from pyevtk.hl import pointsToVTK
import taichi as ti

ti.init(arch=ti.gpu, device_memory_GB=3.0)

from functions import createFilePaths, progressBar, initialize_Cubes, post_process, substep
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


time0 = time.time()

initialize_Cubes()
gui = ti.GUI('Taichi MPM', res=512, background_color=0x112F41, show_gui=False)

filepath, vtkpath = createFilePaths(numerical)

os.getcwd()
if not os.path.exists(filepath):
    os.makedirs(filepath)
if not os.path.exists(vtkpath):
    os.makedirs(vtkpath)


count = 0
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
while (numerical.totalTime < numerical.simulationTime):
    # the 5e-4 here acts as the frame rate
    for s in range(int(5e-4 // numerical.timeStep)):
        substep()
        count = count + 1
        numerical.totalTime += numerical.timeStep

    progressBar(numerical.totalTime, numerical.simulationTime)
    post_process(numerical, particle, gui, vtkpath, filepath)

time1 = time.time()
print('Run Time:', time1 - time0)
