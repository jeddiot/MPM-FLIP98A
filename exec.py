import numpy as np
import os as os
import time
import taichi as ti

ti.init(arch=ti.gpu, device_memory_GB=3.0)

from functionsConfidential import createFilePaths, progressBar, initialize_Cubes, post_process, substep
from config import NumericalSettings, PhysicalQuantities, GravityField
from fields import ParticleFields, GridFields, StabilizationFields, ProjectionFields, PenaltyMethodFields


physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
gravitational = GravityField(numerical, physical)
particle = ParticleFields(numerical.numParticles, numerical.valueType)
grid = GridFields(numerical.numGrids, numerical.valueType)
stability = StabilizationFields(numerical.numGrids, numerical.valueType)
projection = ProjectionFields(numerical.numGrids, numerical.valueType)
penalty = PenaltyMethodFields(numerical.numCells, numerical.valueType)


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
while (numerical.currentTime < numerical.simulationTime):
    # the 5e-4 here acts as the frame rate
    for s in range(int(5e-4 // numerical.timeStep)):
        substep()
        count += 1
        numerical.currentTime += numerical.timeStep

    progressBar(numerical.currentTime, numerical.simulationTime)
    post_process(numerical.numParticles, gui, vtkpath, filepath)

time1 = time.time()
print('Run Time:', time1 - time0)
