# Overview

This project demonstrates the fundamental principles of the Material Point Method (MPM) with a focus on the Concurrent Material Point Method (Concurrent MPM), as developed in my master's thesis. The implementation draws inspiration from Taichi, a computational framework created by Prof. Yuanming Hu et al. at MIT, and integrates the Affine Particle-In-Cell (APIC) method (Jiang et al., 2015) with the Fluid Implicit Particle (FLIP) method.

This work was done in collaboration with **Prof. T.-H. Huang**. We intend to publish our work in 2025. You can find more about his work on his [personal website](https://huan0652.wixsite.com/thhuang).

## Features

- **Updated Lagrangian Mechanics:** Decomposes acceleration, velocity, and position updates for improved simulation efficiency.
- **Material Mixing:** Demonstrates the blending of APIC and FLIP for versatile material behavior.
- **Taichi-Powered Simulation:** Utilizes Taichi's capabilities for high-performance numerical computation.

## Note

Due to laboratory policies, the full source code of the Concurrent MPM cannot be shared. However, this repository includes a simplified demo showcasing the core concepts and techniques.

## References

- **Taichi Framework:** [https://taichi.graphics](https://taichi.graphics)  
- **APIC Method:** Jiang et al. (2015), *The Affine Particle-In-Cell Method*  

Feel free to explore the provided demo and experiment with the Taichi-based implementation.
