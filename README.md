# Overview

This project demonstrates the fundamental principles of the Material Point Method (MPM) with a focus on the Concurrent Material Point Method (Concurrent MPM), as developed in my master's thesis. The implementation draws inspiration from Taichi, a computational framework created by Prof. Yuanming Hu et al. at MIT, and integrates various velocity solvers.

This work was done in collaboration with **Prof. T.-H. Huang**. You can find more about his work on his [personal website](https://huan0652.wixsite.com/thhuang).

## Confidentiality Warning

The file ```functionsConfidential.py``` is not uploaded due to laboratory policy regarding confidential research materials. This file contains proprietary implementations that cannot be publicly shared. However, this repository includes a simplified demo showcasing the core concepts and techniques. For more information, please contact National Tsing Hua University, Extreme Event Computation Lab.

## Initial Validation

We first validate our concept in the C++ source code using Taichi. The result yields a stable solution as shown below.

<img src="output.gif" width="400" height="400" style="display: block; margin-left: auto; margin-right: auto;">

We them proceed to develop our methods in Python for better maintenance.

## Features

- **Updated Lagrangian Mechanics:** Decomposes acceleration, velocity, and position updates for improved simulation efficiency.
- **Material Mixing:** Demonstrates the blending of different velocity solvers for versatile material behavior.
- **Taichi-Powered Simulation:** Utilizes Taichi's capabilities for high-performance numerical computation.

## References

- **Taichi Framework:** [https://taichi.graphics](https://taichi.graphics)  
- **APIC Method:** Jiang et al. (2015), *The Affine Particle-In-Cell Method*  

Feel free to explore the provided demo and experiment with the Taichi-based implementation.
