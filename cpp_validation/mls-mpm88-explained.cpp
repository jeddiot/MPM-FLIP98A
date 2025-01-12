#define TC_IMAGE_IO // Uncomment for image exporting functionality
#include "taichi.h"

using namespace taichi;
using Vec = Vector2;
using Mat = Matrix2;

const int window_size = 800; // gui window size
const int num_grid = 80; // Grid resolution (cells)

const real dt = 1e-4_f;
const real frame_dt = 1e-3_f;
const real dx = 1.0_f / num_grid;
const real inv_dx = 1.0_f / dx;

// material point properties
const auto mass_p = 1.0_f; // mass
const auto vol_p = 1.0_f; // volume
const auto hardening = 1_f; // 10_f; // snow hardening factor
const auto youngsMod = 1e2_f; // 1e4_f; // Young's Modulus
const auto poissonRatio = 0.499_f; //0.2_f; // Poisson ratio
const bool plastic = true;

// lame parameters
const real mu_0 = youngsMod / (2 * (1 + poissonRatio));
const real lambda_0 = youngsMod * poissonRatio / ((1 + poissonRatio) * (1 - 2 * poissonRatio));

struct Particle {
    Vec x, v; // position, velocity
    Mat F; // deformation gradient
    Mat C; // velocity gradient from APIC
    real Jp; // determinant of the deformation gradient (i.e. volume)
    int c; // color

    Particle(Vec x, int c, Vec v=Vec(0)) :
        x(x),
        v(v),
        F(1),
        C(0),
        Jp(1),
        c(c) {}
};

std::vector<Particle> particles;

// Vector3: [velocity_x, velocity_y, mass]
Vector3 grid[num_grid + 1][num_grid + 1];

void advance(real dt) {
    std::memset(grid, 0, sizeof(grid)); // reset grids

    // P2G
    for (auto &p : particles) {
        // element-wise floor
        Vector2i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();

        Vec fx = p.x * inv_dx - base_coord.cast<real>();

        // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
        Vec w[3] = {
            Vec(0.5) * sqr(Vec(1.5) - fx),
            Vec(0.75) - sqr(fx - Vec(1.0)),
            Vec(0.5) * sqr(fx - Vec(0.5))
        };

        // Compute current Lamé parameters [http://mpm.graphics Eqn. 86]
        auto e = std::exp(hardening * (1.0f - p.Jp));
        auto mu = mu_0 * e;
        auto lambda = lambda_0 * e;

        // Current volume
        real J = determinant(p.F);

        // Polar decomposition for fixed corotated model
        Mat r, s;
        polar_decomp(p.F, r, s);

        // [http://mpm.graphics Paragraph after Eqn. 176]
        real Dinv = 4 * inv_dx * inv_dx;
        // [http://mpm.graphics Eqn. 52]
        auto PF = (2 * mu * (p.F-r) * transposed(p.F) + lambda * (J-1) * J);

        // Cauchy stress times dt and inv_dx
        auto stress = - (dt * vol_p) * (Dinv * PF);

        // Fused APIC momentum + MLS-MPM stress contribution
        // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
        // Eqn 29
        auto affine = stress + mass_p * p.C;

        // P2G
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                auto dpos = (Vec(i, j) - fx) * dx;
                // Translational momentum
                Vector3 mass_x_velocity(p.v * mass_p, mass_p);
                grid[base_coord.x + i][base_coord.y + j] += (
                w[i].x*w[j].y * (mass_x_velocity + Vector3(affine * dpos, 0))
                );
            }
        }
    }

    // grid update
    for(int i = 0; i <= num_grid; i++) {
        for(int j = 0; j <= num_grid; j++) {
            auto &g = grid[i][j];
            // No need for epsilon here
            if (g[2] > 0) {
                // Normalize by mass
                g /= g[2];
                // Gravity
                g += dt * Vector3(0, -200, 0);

                // boundary thickness
                real boundary = 0.05;
                // Node coordinates
                real x = (real) i / num_grid;
                real y = real(j) / num_grid;

                // Sticky boundary
                if (x < boundary || x > 1-boundary || y > 1-boundary) {
                    g = Vector3(0);
                }
                // Separate boundary
                if (y < boundary) {
                    g[1] = std::max(0.0f, g[1]);
                }
            }
        }
    }

  // G2P
    for (auto &p : particles) {
        // element-wise floor
        Vector2i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        Vec w[3] = {
                    Vec(0.5) * sqr(Vec(1.5) - fx),
                    Vec(0.75) - sqr(fx - Vec(1.0)),
                    Vec(0.5) * sqr(fx - Vec(0.5))
        };

        p.C = Mat(0);
        p.v = Vec(0);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                auto dpos = (Vec(i, j) - fx);
                auto grid_v = Vec(grid[base_coord.x + i][base_coord.y + j]);
                auto weight = w[i].x * w[j].y;

                p.v += weight * grid_v; // velocity
                p.C += 4 * inv_dx * Mat::outer_product(weight * grid_v, dpos); // APIC C
            }
        }

        // Advection
        p.x += dt * p.v;

        // MLS-MPM F-update
        auto F = (Mat(1) + dt * p.C) * p.F;

        Mat svd_u, sig, svd_v;
        svd(F, svd_u, sig, svd_v);

        // Snow Plasticity
        for (int i = 0; i < 2 * int(plastic); i++) {
            sig[i][i] = taichi::clamp(sig[i][i], 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
        }

        real oldJ = determinant(F);
        F = svd_u * sig * transposed(svd_v);

        real Jp_new = taichi::clamp(p.Jp * oldJ / determinant(F), 0.6f, 20.0f);

        p.Jp = Jp_new;
        p.F = F;
    }
}

// // Seed particles with position and color
// void add_object(Vec center, int c) {
//   // Randomly sample 1000 particles in the square
//   for (int i = 0; i < 1000; i++) {
//     particles.push_back(Particle((Vec::rand()*2.0f-Vec(1))*0.08f + center, c));
//   }
// }

// Add a large block at the bottom and a small block above it
void add_object(Vec center) {
    // Add a large block at the bottom (larger and fixed)
    for (int i = 0; i < 3000; i++) {
        particles.push_back(Particle((Vec::rand() * 2.0f - Vec(1)) * 0.08f + Vec((0.05f + 0.08f), (0.05f + 0.08f)), 0x2986CC));
    }

    // // Add a small block on top to simulate impact
    // for (int i = 0; i < 100; i++) {
    //     particles.push_back(Particle((Vec::rand() * 2.0f - Vec(1)) * 0.02f + center, 0xED553B));
    // }
}

int main() {
    GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
    auto &canvas = gui.get_canvas();

    add_object(Vec(0.5,0.5));
    // add_object(Vec(0.45,0.65), 0xF2B134);
    // add_object(Vec(0.55,0.85), 0x068587);

    int frame = 0;

    // Main Loop
    for (int step = 0; step < 2500; step++) {
        advance(dt); // advance simulation
        // Visualize frame
        if (step % int(frame_dt / dt) == 0) {
            canvas.clear(0x112F41); // clear background
            canvas.rect(Vec(0.04), Vec(0.96)).radius(2).color(0x52BFBF).close(); // boundary
            for (auto p : particles) { // Particles
                canvas.circle(p.x).radius(2).color(p.c);
            }
            gui.update(); // update image
            canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", frame++)); // write to disk
        }
    }
}

/* -----------------------------------------------------------------------------
** Reference: Y. Hu et al., A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling (SIGGRAPH 2018)

run commands:
g++ mls-mpm88-explained.cpp -std=c++14 -lgdi32 -lpthread -O3 -o mls-mpm     
.\mls-mpm.exe
ffmpeg -framerate 60 -i %05d.png -c:v libx264 -pix_fmt yuv420p output.mp4
----------------------------------------------------------------------------- */
