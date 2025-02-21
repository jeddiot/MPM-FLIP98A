import taichi as ti


class NumericalSettings:
    def __init__(self):
        # Numerical Parameters
        self.dimension = 2  # Problem dimension
        self.epsilon = 1e-15  # Numerical tolerance
        self.simulationTime = 0.1  # Total simulation time
        self.totalTime = 0.0  # Initialize simulation time
        self.timeStep = 1e-5  # Timestep
        self.penalty = 1e6  # Penalty parameter
        self.pressureMixingRatio = 1  # 0: point-wise pressure, 1: mixed formulation pressure
        self.sizeScale = 1.0
        self.flipBlendParameter = 0.0  # FLIP blend parameter (0.0 = full APIC, 1.0 = full FLIP)

        # Particle Grid Settings
        self.numParticlesX = 80  # Number of particles in x direction
        self.numParticlesY = 40  # Number of particles in y direction
        self.numParticles = self.numParticlesX * self.numParticlesY  # Total number of particles

        self.domainLength = 6.0 / self.sizeScale  # Length of the domain [m]
        self.fluidWidth = 4.0 / self.sizeScale  # Width of liquid square [m]
        self.fluidHeight = 2.0 / self.sizeScale  # Height of liquid square [m]
        self.initialParticleVolume = (self.fluidWidth * self.fluidHeight) / self.numParticles  # Initial volume per particle

        self.numGrids = 75  # simulation domain (71) + left and right boundaries (4)
        self.numCells = self.numGrids - 1
        self.gridSpacing = self.domainLength / float(self.numCells - 4)  # Grid spacing
        self.inverseGridSpacing = 1 / self.gridSpacing

        self.kernelSupportSizeNormalized = 1.5  # RK normalized kernel function support size
        self.kernelSupportSize = self.kernelSupportSizeNormalized * self.gridSpacing  # RK kernel function support size
        # Max number of 1D grid nodes in the support of each particle
        self.nodeCount = int(self.kernelSupportSize * self.inverseGridSpacing * 2 + self.epsilon)
        self.gridNodeShift = float(self.kernelSupportSize * self.inverseGridSpacing - 1.0)  # Base shift value


class PhysicalQuantities:
    def __init__(self):
        # Physical Parameters
        self.dynamicViscosity = 1e-3  # Dynamic viscosity [Pa·s]
        self.poissonsRatio = 4.999e-1  # Poisson's ratio (unitless)
        self.bulkModulus = 2e9  # Bulk modulus [Pa]
        self.youngsModulus = self.bulkModulus * 2 * (1 - self.poissonsRatio)  # Young's modulus [Pa]
        self.shearModulus = self.bulkModulus * (1 - self.poissonsRatio) / (1 + self.poissonsRatio)  # Shear modulus [Pa]
        self.particleDensity = 997.5  # Particle density [kg/m³]
        self.gravity = -9.81  # Gravity [m/s²]


class GravityField:
    def __init__(self, fluidWidth, fluidHeight, particleDensity, gravity):
        # Gravity field using Taichi
        self.gravityField = ti.Vector.field(2, dtype=ti.f64, shape=())  # Gravity vector field
        self.gravityField[None] = [0, fluidWidth * fluidHeight * particleDensity * gravity]  # Set initial gravity direction


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
