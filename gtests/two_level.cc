#include <deal.II/base/function_lib.h>   // For Functions::ZeroFunction
#include <deal.II/base/quadrature_lib.h> // Added for QGauss

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h> // Added for AffineConstraints
#include <deal.II/lac/arpack_solver.h>
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

TEST(EigenValues, OutputEigenvalues)
{
  const unsigned int       dim      = 2;
  const std::string        filename = SOURCE_DIR "/prms/01_eigenvalues.prm";
  TestBenchParameters<dim> parameters(filename);
  parameters.output_directory = SOURCE_DIR "/output";
  TestBench<dim> test_bench(parameters);
  test_bench.initialize();

  // 1. Setup constraints and assemble matrices
  AffineConstraints<double> constraints;
  constraints.clear();
  const Functions::ZeroFunction<dim> zero_function(
    test_bench.fe->n_components());
  const auto b_ids = test_bench.triangulation.get_boundary_ids();
  for (const auto &b_id : b_ids)
    {
      VectorTools::interpolate_boundary_values(test_bench.dof_handler,
                                               b_id,
                                               zero_function,
                                               constraints);
    }
  constraints.close();

  DynamicSparsityPattern dsp(test_bench.dof_handler.n_dofs(),
                             test_bench.dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(test_bench.dof_handler, dsp, constraints);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);


  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> mass_matrix;

  SparseDirectUMFPACK stiffness_inverse;

  stiffness_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);

  QGauss<dim> quadrature_formula(test_bench.fe->degree + 1);

  MatrixCreator::create_laplace_matrix(test_bench.dof_handler,
                                       quadrature_formula,
                                       stiffness_matrix,
                                       {},
                                       constraints);
  MatrixCreator::create_mass_matrix(
    test_bench.dof_handler, quadrature_formula, mass_matrix, {}, constraints);

  stiffness_inverse.initialize(stiffness_matrix);

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

      ArpackSolver solver(control);

      solver.solve(stiffness_matrix,
                   mass_matrix,
                   stiffness_inverse,
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