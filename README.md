# High performance solution of partial differential equations

[![GitHub CI](https://github.com/luca-heltai/hpspde/actions/workflows/gtest.yaml/badge.svg)](https://github.com/luca-heltai/hpspde/actions/workflows/gtest.yaml)

## Abstract

Partial differential equations (PDEs) are fundamental tools for modeling a wide range of scientific and engineering phenomena. This PhD course focuses on developing high-performance computational methods for solving real world PDEs.
The course will explore the following key areas:

- Finite Element Method (FEM): We will quickly review the theoretical foundation of FEM, a powerful technique for discretizing PDEs into a system of algebraic equations.
- deal.II Library: The course will leverage the capabilities of deal.II, a high-performance finite element library, for efficient implementation of FEM discretizations. All the theory is also presented in the context of deal.II, which is a C++ library for solving PDEs using the finite element method. The library provides a wide range of features, including adaptive mesh refinement, parallel computing, and support for various finite element types.
- Domain Decomposition Methods: We will explore strategies for parallelizing FEM computations by decomposing the computational domain into subdomains, enabling efficient utilization of high-performance computing resources.
- Parallel Linear Algebra Techniques: Techniques for solving large-scale linear systems arising from FEM discretizations on parallel architectures will be a key focus. This includes exploring libraries like PETSc or Trilinos, in conjunction with deal.II.
- Matrix-Free Geometric Multigrid Methods: We will investigate matrix-free geometric multigrid methods, which are  powerful iterative solvers that exploit the geometric structure of the problem to achieve optimal scaling properties, with negligible storage requirements.

By combining these elements, the course equips students with the knowledge and skills necessary to develop and implement high-performance solutions for complex PDEs on modern computing platforms. Students will gain hands-on experience through programming exercises and case studies, enabling them to tackle real-world scientific and engineering challenges.
Prerequisites: This course assumes a strong foundation in applied mathematics, including numerical analysis, linear algebra, and partial differential equations. Programming experience (C++ preferred) is also beneficial.

Reference books:

- Theory and Practice of Finite Elements, 2004, Alexander Ern, Jean-Luc Guermond
- Numerical Linear Algebra for High-Performance Computers, 1998, Jack J. Dongarra, Iain S. Duff, Danny C. Sorensen, Henk A. van der Vorst
- Domain Decomposition Methods - Algorithms and Theory, 2005, Andrea Toselli, Olof B. Widlund
- The Analysis of Multigrid Methods, 2000, James H. Bramble Xuejun Zhang

# Schedule

| Date       | Topic                                                                 |
|------------|----------------------------------------------------------------------|
| 2025-04-07 | Preliminary meeting for the course                                   |
| 2025-04-08 | Crash course on to git, docker, github actions, continuous integration, and remote development with Visual Studio Code                               |

# Course material

- Git tutorial: <https://javedali99.github.io/git-tutorial/slides.html>
- Docker tutorial: <https://www.docker.com/101-tutorial>
- deal.II web page: <https://www.dealii.org/>
- Remote container development with Visual Studio Code: <https://code.visualstudio.com/docs/remote/containers>
- Continuous integration with GitHub actions: <https://docs.github.com/en/actions/learn-github-actions/introduction-to-github-actions>

# Participants

Stefano