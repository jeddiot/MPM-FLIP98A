import taichi as ti


class PhysicalQuantities:
    def __init__(self):
        self.dynamicViscosity = 1e-3  # unit: [Pa·s]
        self.poissonsRatio = 4.999e-1  # unitless
        self.bulkModulus = 2e9  # unit: [Pa]
        self.youngsModulus = self.bulkModulus * 2 * (1 - self.poissonsRatio)  # unit: [Pa]
        self.shearModulus = self.bulkModulus * (1 - self.poissonsRatio) / (1 + self.poissonsRatio) # unit: [Pa]
        self.particleDensity = 997.5  # unit: [kg/m³]
        self.gravity = -9.81  # unit: [m/s²]


class NumericalSettings:
    def __init__(self, physical: PhysicalQuantities):

        self.valueType = ti.f64
        self.switch_vt_I_APIC = True  # True: velocity APIC, False: velocity PIC
        self.switch_overlineF = False  # F-Bar pressure stabilization
        self.switch_penaltyEBC = False
        self.switch_kernelFunction = True  # bspline, tent
        self.pressureMixingRatio = 0  # 0: point-wise pressure, 1: mixed formulation pressure
        self.sizeScale = 1
        self.flipBlendParameter = 1  # (0.0 = full APIC, 1.0 = full FLIP)

        self.dimension = int(2)
        self.numericalTolerance = 1e-15
        self.simulationTime = 0.1
        self.currentTime = float(0)  # Initialize simulation time
        self.timeStep = 1e-5
        self.numParticlesX = 80
        self.numParticlesY = 40
        self.numParticles = self.numParticlesX * self.numParticlesY
        self.domainLength = 6 / self.sizeScale  # [m]
        self.fluidWidth = 4 / self.sizeScale  # [m]
        self.fluidHeight = 2 / self.sizeScale  # [m]
        self.initialParticleVolume = (self.fluidWidth * self.fluidHeight) / self.numParticles
        self.numGrids = 75 # domain (71) + boundaries (2 grid nodes on each boundary)
        self.numCells = self.numGrids - 1
        self.gridSpacing = self.domainLength / float(self.numCells - 4) # 2 grid nodes on each boundary, left + right -> 4
        self.inverseGridSpacing = 1 / self.gridSpacing
        self.kernelSupportSizeNormalized = 1.5
        self.kernelSupportSize = self.kernelSupportSizeNormalized * self.gridSpacing
        self.maxNumof1DgridNodesWithinSupport = int(self.kernelSupportSize * self.inverseGridSpacing * 2 + self.numericalTolerance)
        self.gridNodeShift = float(self.kernelSupportSize * self.inverseGridSpacing - 1.0)
        self.normalizedPenaltyParameter = 1e6
        self.penaltyParameter = self.normalizedPenaltyParameter * physical.particleDensity * self.gridSpacing**2


class GravityField:
    def __init__(self, numerical: NumericalSettings, physical: PhysicalQuantities):
        self.gravityField = ti.Vector.field(2, dtype=numerical.valueType, shape=(1, ))
        self.gravityField[0] = [0, numerical.fluidWidth * numerical.fluidHeight * physical.particleDensity * physical.gravity]


class SimulationConfig:
    def __init__(self):
        # Initialize all classes
        self.numericalSettings = NumericalSettings()
        self.physicalQuantities = PhysicalQuantities()
        self.gravityField = GravityField(self.numericalSettings.fluidWidth,
                                         self.numericalSettings.fluidHeight,
                                         self.physicalQuantities.particleDensity,
                                         self.physicalQuantities.gravity)

    def __repr__(self):
        return f"SimulationConfig({vars(self.numericalSettings)}, {vars(self.physicalQuantities)}, {vars(self.gravityField)})"


physical = PhysicalQuantities()
numerical = NumericalSettings(physical)
gravatational = GravityField(numerical, physical)
