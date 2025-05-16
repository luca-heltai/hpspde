#include <deal.II/base/function_lib.h>   // For Functions::ZeroFunction
#include <deal.II/base/quadrature_lib.h> // Added for QGauss

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h> // Added for AffineConstraints
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>

#include <algorithm> // For std::min
#include <fstream>
#include <set> // For std::set
#include <string>
#include <vector>

#include "test_bench.h"


using namespace dealii;

TEST(EigenValues, OutputEigenvalues01)
{
  const unsigned int       dim      = 2;
  const std::string        filename = SOURCE_DIR "/prms/01_eigenvalues.prm";
  TestBenchParameters<dim> parameters(filename);
  parameters.output_directory = SOURCE_DIR "/output";
  TestBench<dim> test_bench(parameters);
  test_bench.initialize();

  Poisson<dim> poisson(test_bench.dof_handler,
                       Functions::ZeroFunction<dim>(
                         test_bench.fe->n_components()));

  // The old MatrixTools::apply_boundary_values calls and dummy vectors are
  // removed.

  // 2. Solve the eigenvalue problem A u = lambda M u
  const unsigned int n_dofs = test_bench.dof_handler.n_dofs();
  const unsigned int n_eigenvalues_to_compute =
    std::min(5U, (n_dofs > 1 ? n_dofs - 1 : 1U));

  std::vector<std::complex<double>> computed_eigenvalues(
    n_eigenvalues_to_compute);
  std::vector<Vector<double>> computed_eigenvectors(n_eigenvalues_to_compute *
                                                    2);
  for (auto &ev : computed_eigenvectors)
    ev.reinit(n_dofs);

  if (n_dofs > 0 && n_eigenvalues_to_compute > 0)
    {
      // Use ArpackSolver::Control to initialize the solver
      SolverControl control;

      ArpackSolver::AdditionalData data(15, // Number of Arnoldi/Lanczos vectors
                                        ArpackSolver::smallest_magnitude,
                                        true); // Symmetric problem

      ArpackSolver solver(control, data);

      solver.solve(poisson.stiffness_matrix,
                   poisson.mass_matrix,
                   poisson.stiffness_inverse,
                   computed_eigenvalues,
                   computed_eigenvectors);

      // 3. Output eigenvalues to a file
      std::ofstream eigenvalues_file(parameters.output_directory +
                                     "/eigenvalues.txt");
      eigenvalues_file << "Computed eigenvalues:"
                       << std::endl;                 // Style: use std::endl
      for (const auto &value : computed_eigenvalues) // Style: range-based for
        {
          eigenvalues_file << value << std::endl; // Style: use std::endl
        }
      eigenvalues_file.close();

      // 4. Output eigenvectors to VTU files
      for (unsigned int i = 0; i < computed_eigenvectors.size(); ++i)
        {
          test_bench.output_results(computed_eigenvectors[i],
                                    "eigenvector_" + std::to_string(i));
        }
    }
  else
    {
      std::cout << "Skipping eigenvalue computation due to n_dofs=" << n_dofs
                << " or n_eigenvalues_to_compute=" << n_eigenvalues_to_compute
                << std::endl; // Style: use std::endl
    }
}


TEST(EigenValues, Smoother)
{
  const unsigned int       dim      = 2;
  const std::string        filename = SOURCE_DIR "/prms/smoother.prm";
  TestBenchParameters<dim> parameters(filename);
  parameters.output_directory = SOURCE_DIR "/output";
  parameters.output_file_name = "smoother";
  TestBench<dim> test_bench(parameters);
  test_bench.initialize();

  Poisson<dim> poisson(test_bench.dof_handler,
                       Functions::ZeroFunction<dim>(
                         test_bench.fe->n_components()));

  auto exact_solution = test_bench.make_vector();
  auto rhs            = test_bench.make_vector();

  // Fill solution with random values
  for (auto &value : exact_solution)
    value = Utilities::generate_normal_random_number(1.0, 0.1);
  poisson.constraints.distribute(exact_solution);

  test_bench.output_results(exact_solution, "0");

  // Create a linear operator for A
  auto A  = linear_operator(poisson.stiffness_matrix);
  auto M  = linear_operator(poisson.mass_matrix);
  auto Id = identity_operator(A);

  rhs = A * exact_solution;

  // Jacobi preconditioner


  // Smoother definition
  auto Pinv = parameters.omega * Id;

  auto solution = test_bench.make_vector();
  auto residual = test_bench.make_vector();
  auto tmp      = test_bench.make_vector();

  // x^n+1 = x^n + omega * (b-A x^n)
  // e^n+1 = e^n - omega * A e^n

  std::cout << "Solving with smoother parameter " << parameters.omega << " for "
            << parameters.smoothing_steps << " steps." << std::endl;
  solution = 0;
  for (unsigned int i = 1; i < parameters.smoothing_steps; ++i)
    {
      // Start with the residual
      residual = rhs - A * solution;

      solution += Pinv * residual;

      // poisson.constraints.distribute(solution);

      tmp = solution;
      tmp -= exact_solution;

      // Output the solution
      test_bench.output_results(solution, std::to_string(i));
      std::cout << "Iteration " << i
                << ": max residual = " << residual.l2_norm()
                << ", error: " << tmp.l2_norm() << std::endl;
    }


  // Fill rhs with the expected rhs:
}