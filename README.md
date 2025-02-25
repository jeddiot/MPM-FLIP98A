# Developing Stabilized Material Point Method for Free Surface Flow Modeling

## Overview

This project is part of **Jedd (Cheng-Chun Yang)'s** master's thesis titled *"Developing Stabilized Material Point Method for Free Surface Flow Modeling"* (開發穩態物質點法應用於自由表面流建模), advised by **Prof. Tsung-Hui Huang** at National Tsing Hua University, 2024. The work is set to be published in 2025. For more information about this work, please visit [Prof. Tsung-Hui Huang's personal website](https://huan0652.wixsite.com/thhuang).

### Key Files

- `config.py`: Specifies the numerical settings for the simulation.
- `fields.py`: Declares the Taichi fields used in the simulation.
- `exec.py`: The entry point where the iteration of the simulation takes place.
- `functionsConfidential.py`: Contains the innovative algorithm for the **Concurrent Material Point Method (MPM)**, specifically demonstrated in the `subStep()` function.

This simulation is powered by the **Taichi runtime environment**, a high-performance computational framework developed by **Prof. Yuanming Hu** and colleagues. The implementation of the algorithm is inspired by the **Affine Particle-In-Cell (APIC)** method (Jiang et al., 2015) and the **Fluid Implicit Particle (FLIP)** method.

### Contributions

- Innovative algorithm under material point method scheme for computational fluid dynamics.
- A practical programming implementation using Taichi's computational capabilities.

We hope to contribute to the community's understanding and application of stabilized numerical methods in fluid dynamics.

## Confidentiality Warning

The file ```functionsConfidential.py``` is not uploaded due to laboratory policy regarding confidential research materials. This file contains proprietary implementations that cannot be publicly shared. However, this repository includes a simplified demo showcasing the core concepts and techniques. For more information, please contact National Tsing Hua University, Extreme Event Computation Lab.

## Demonstration

We first validate our concept in the c++ source code using Taichi. The simulation yields a stable solution, as shown below.

<img src="output.gif" width="400" height="400" style="display: block; margin-left: auto; margin-right: auto;">

We them proceed to develop our methods in Python for better maintenance.

(Will be uploaded in the near future.)

## Features

- **Updated Lagrangian Mechanics:** Decomposes acceleration, velocity, and position updates for improved simulation efficiency.
- **Material Mixing:** Demonstrates the blending of different velocity solvers for versatile material behavior.
- **Taichi-Powered Simulation:** Utilizes Taichi's capabilities for high-performance numerical computation.

## References

- **Taichi Framework:** [https://taichi.graphics](https://taichi.graphics)  
- **APIC Method:** Jiang et al. (2015), *The Affine Particle-In-Cell Method*  

Feel free to explore the provided demo and experiment with the Taichi-based implementation.