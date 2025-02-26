import time
from pyevtk.hl import pointsToVTK
import taichi as ti

ti.init(arch=ti.gpu, device_memory_GB=3.0)

from functionsConfidential import createFilePaths, progressBar, initialize_Cubes, post_process, subStep
from config import NumericalSettings, PhysicalQuantities, GravityField
from fields import ParticleFields, GridFields, StabilizationFields, ProjectionFields, PenaltyMethodFields


physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
particle = ParticleFields(numerical.numParticles, numerical.valueType)


timeSimulationBegin = time.time()

initialize_Cubes()
gui = ti.GUI('Taichi MPM', res=512, background_color=0x112F41, show_gui=False)

filepath, vtkpath = createFilePaths(numerical)

count = 0
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
while (numerical.totalTime < numerical.simulationTime):
    num_substeps = int(numerical.frameRate // numerical.timeStep)
    for s in range(num_substeps):
        subStep()
        count += 1
        numerical.totalTime += numerical.timeStep

    progressBar(numerical.totalTime, numerical.simulationTime)
    post_process(numerical.numParticles, gui, vtkpath, filepath, num_substeps, count)

timeSimulationEnd = time.time()
print('Run Time:', timeSimulationEnd - timeSimulationBegin)
