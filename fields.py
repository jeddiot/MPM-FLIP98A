import taichi as ti


class ParticleFields:
    def __init__(self, num_p):
        self.position = ti.Vector.field(2, dtype=ti.f64, shape=num_p)
        self.velocity = ti.Vector.field(2, dtype=ti.f64, shape=num_p)
        self.velocity_gradient = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p)
        self.deformation_gradient = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p)
        self.determinant_of_deformation_gradient = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p)
        self.stress = ti.Matrix.field(2, 2, dtype=ti.f64, shape=num_p)
        self.material_id = ti.field(dtype=int, shape=num_p)
        self.volume = ti.field(dtype=ti.f64, shape=num_p)
        self.mass = ti.field(dtype=ti.f64, shape=num_p)
        self.partitionofUnity = ti.field(dtype=ti.f64, shape=num_p)
        self.consistency = ti.field(dtype=ti.f64, shape=num_p)
        self.consistency_dx = ti.field(dtype=ti.f64, shape=num_p)
        self.consistency_dy = ti.field(dtype=ti.f64, shape=num_p)
        self.pressure = ti.field(dtype=ti.f64, shape=num_p)
        self.divergenceofVelocity = ti.field(dtype=ti.f64, shape=num_p)
        self.particleDensity = ti.field(dtype=ti.f64, shape=num_p)


class GridFields:
    def __init__(self, num_g):
        self.velocity_grid = ti.Vector.field(2, dtype=ti.f64, shape=(num_g, num_g))
        self.velocity_grid_initial = ti.Vector.field(2, dtype=ti.f64, shape=(num_g, num_g))
        self.mass_grid = ti.Matrix.field(2, 2, dtype=ti.f64, shape=(num_g, num_g))
        self.volume_grid = ti.field(dtype=ti.f64, shape=(num_g, num_g))
        self.pressure_grid = ti.field(dtype=ti.f64, shape=(num_g, num_g))


class StabilizationFields:
    def __init__(self, num_g):
        self.volume_0 = ti.field(dtype=ti.f64, shape=(num_g - 1, num_g - 1))
        self.cell = ti.field(dtype=ti.f64, shape=(num_g - 1, num_g - 1))


class ProjectionFields:
    def __init__(self, num_g):
        self.divergence_velocity_numerator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
        self.divergence_velocity_denominator = ti.field(dtype=ti.f64, shape=(num_g, num_g))
        self.divergence_velocity = ti.field(dtype=ti.f64, shape=(num_g, num_g))


class PenaltyMethodFields:
    def __init__(self, num_cell):
        self.left_boundary = ti.Vector.field(2, dtype=ti.f64, shape=num_cell - 4)
        self.right_boundary = ti.Vector.field(2, dtype=ti.f64, shape=num_cell - 4)
        self.bottom_boundary = ti.Vector.field(2, dtype=ti.f64, shape=num_cell - 4)
        self.top_boundary = ti.Vector.field(2, dtype=ti.f64, shape=num_cell - 4)
