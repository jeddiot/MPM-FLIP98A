import time
import taichi as ti
ti.init(arch=ti.gpu, device_memory_GB=3.0)
from config import NumericalSettings, PhysicalQuantities
from functionsConfidential import createFilePaths, progressBar, initialization, post_process, subStep

physical = PhysicalQuantities()
numerical = NumericalSettings(physical)

timeSimulationBegin = time.time()

initialization()

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
