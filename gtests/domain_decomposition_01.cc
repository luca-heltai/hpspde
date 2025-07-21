#include <deal.II/base/function_lib.h> // For Functions::ZeroFunction
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h> // Added for QGauss

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h> // Added for AffineConstraints
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <exception>

#include "test_bench.h"

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


#include <gtest/gtest.h>

#include <algorithm> // For std::min
#include <fstream>
#include <string>
#include <vector>

#include "test_bench.h"
#include "two_level_multigrid.h"

using namespace dealii;

TEST(DomainDecomposition, MPI_BasicTest)
{
  const unsigned int                        dim = 2;
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0, 1, false);
  triangulation.refine_global(4);

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  LA::MPI::Vector totally_distributed_solution(locally_owned_dofs,
                                               MPI_COMM_WORLD);

  LA::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "DoFHandler: " << dof_handler.n_dofs() << std::endl;
      std::cout << "Locally owned DoFs: " << locally_owned_dofs.n_elements()
                << std::endl;
      std::cout << "Locally relevant DoFs: "
                << locally_relevant_dofs.n_elements() << std::endl;

      std::cout << "Locally relevant DoFs: ";
      locally_relevant_dofs.print(std::cout);

      std::cout << "Locally owned DoFs: ";
      locally_owned_dofs.print(std::cout);

      totally_distributed_solution[100] = 1.0;

      totally_distributed_solution[180] = 1.0;
    }
  totally_distributed_solution.compress(VectorOperation::insert);
  locally_relevant_solution = totally_distributed_solution;
  std::cout << "totally_distributed_solution norm:"
            << totally_distributed_solution.l1_norm() << std::endl;

  ASSERT_EQ(locally_relevant_solution.l1_norm(),
            totally_distributed_solution.l1_norm());

  ASSERT_EQ(locally_relevant_solution[159], 0.0);
}


TEST(DomainDecomposition, SplitDomains)
{
  const unsigned int dim      = 2;
  const std::string  filename = SOURCE_DIR "/prms/01_domain_decomposition.prm";
  TestBenchParameters<dim> parameters(filename);
  parameters.output_directory = SOURCE_DIR "/output";
  TestBench<dim> test_bench(parameters);

  parameters.renumbering = {"subdomain"};

  // Manually run some of the methods to test domain decomposition
  test_bench.make_grid();

  // Partition the triangulation into 4 subdomains
  unsigned int n_partitions = 4;
  GridTools::partition_triangulation(n_partitions, test_bench.triangulation);

  // This will also renumber the DoFs according to the subdomain
  test_bench.setup_system();

  std::vector<types::subdomain_id> subdomain_association(
    test_bench.dof_handler.n_dofs(), numbers::invalid_subdomain_id);

  DoFTools::get_subdomain_association(test_bench.dof_handler,
                                      subdomain_association);

  std::vector<IndexSet> subdomain_index_sets(
    n_partitions, IndexSet(test_bench.dof_handler.n_dofs()));

  for (unsigned int i = 0; i < test_bench.dof_handler.n_dofs(); ++i)
    subdomain_index_sets[subdomain_association[i]].add_index(i);

  for (auto &id : subdomain_index_sets)
    id.compress();
  BlockIndices block_indices;
  for (unsigned int i = 0; i < n_partitions; ++i)
    {
      std::cout << "Subdomain " << i << ": ";
      subdomain_index_sets[i].print(std::cout);
      std::cout << std::endl;
      block_indices.push_back(subdomain_index_sets[i].n_elements());
    }

  BlockDynamicSparsityPattern dsp(block_indices, block_indices);

  AffineConstraints<double> constraints;

  VectorTools::interpolate_boundary_values(test_bench.dof_handler,
                                           0,
                                           parameters.exact_solution,
                                           constraints);

  // Create hanging node constraints
  DoFTools::make_hanging_node_constraints(test_bench.dof_handler, constraints);
  constraints.close();

  DoFTools::make_sparsity_pattern(test_bench.dof_handler, dsp, constraints);

  BlockSparsityPattern block_sparsity_pattern;
  block_sparsity_pattern.copy_from(dsp);

  BlockSparseMatrix<double> system_matrix;
  system_matrix.reinit(block_sparsity_pattern);

  MatrixCreator::create_laplace_matrix(test_bench.dof_handler,
                                       QGauss<dim>(2 * test_bench.fe->degree +
                                                   1),
                                       system_matrix,
                                       {},
                                       constraints);

  // Now create linear operators
  auto A = block_operator(system_matrix);

  // And create a block_jacobi_preconditioner
  auto J_A = block_diagonal_operator(system_matrix);

  // Make block solvers
  std::vector<SparseDirectUMFPACK> direct_solvers(n_partitions);
  for (unsigned int i = 0; i < n_partitions; ++i)
    {
      direct_solvers[i].initialize(system_matrix.block(i, i));
      J_A.block(i, i) =
        linear_operator<Vector<double>>(J_A.block(i, i), direct_solvers[i]);
    }

  BlockVector<double> solution(block_indices);
  BlockVector<double> rhs(block_indices);

  VectorTools::create_right_hand_side(test_bench.dof_handler,
                                      QGauss<dim>(2 * test_bench.fe->degree +
                                                  1),
                                      parameters.rhs_function,
                                      rhs);

  // Now create a linear solver
  SolverControl                 solver_control(1000, 1e-12);
  SolverCG<BlockVector<double>> solver(solver_control);
  solver.solve(A, solution, rhs, J_A);
  constraints.distribute(solution);

  std::cout << "Converged in " << solver_control.last_step()
            << " iterations with residual " << solver_control.last_value()
            << std::endl;
}