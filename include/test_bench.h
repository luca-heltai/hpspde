#ifndef TEST_BENCH_H
#define TEST_BENCH_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <string>

using namespace dealii;

template <int dim, int spacedim = dim>
struct TestBenchParameters
{
  TestBenchParameters(const std::string &filename = "test_bench.prm");

  unsigned int initial_refinement        = 1;
  unsigned int n_cycles                  = 1;
  double       omega                     = 1.0;
  unsigned int smoothing_steps           = 10;
  std::string  fe_name                   = "FE_Q(1)";
  std::string  exact_solution_expression = "sin(2*pi*x)*sin(2*pi*y)";
  std::string  grid_name                 = "hyper_cube";
  std::string  grid_arguments            = "0 : 1 : false";
  std::string  output_file_name          = "solution";
  std::string  output_directory          = ".";
  std::vector<std::string> renumbering;

  FunctionParser<spacedim>       exact_solution;
  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};

template <int dim, int spacedim = dim>
class TestBench
{
public:
  TestBench(const TestBenchParameters<dim, spacedim> &parameters);
  void
  initialize();
  void
  make_grid();
  void
  setup_system();

  Vector<double>
  make_vector() const;

  void
  interpolate();
  void
  output_results(const Vector<double> &output_field,
                 const std::string    &suffix) const;

  const TestBenchParameters<dim, spacedim> &par;

  Triangulation<dim, spacedim>                  triangulation;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;
  DoFHandler<dim, spacedim>                     dof_handler;

  Vector<double> solution;
};

template <int dim, int spacedim = dim>
struct Poisson
{
  Poisson(const DoFHandler<dim, spacedim> &dh,
          const Function<spacedim>        &boundary_conditions =
            Functions::ZeroFunction<spacedim>())
  {
    initialize(dh, boundary_conditions);
  }

  void
  initialize(const DoFHandler<dim, spacedim> &dh,
             const Function<spacedim>        &boundary_conditions =
               Functions::ZeroFunction<spacedim>())
  {
    constraints.clear();
    const auto b_ids = dh.get_triangulation().get_boundary_ids();
    for (const auto &b_id : b_ids)
      {
        VectorTools::interpolate_boundary_values(dh,
                                                 b_id,
                                                 boundary_conditions,
                                                 constraints);
      }
    DoFTools::make_hanging_node_constraints(dh, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
    DoFTools::make_sparsity_pattern(dh, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    QGauss<dim> quadrature_formula(dh.get_fe().degree + 1);

    MatrixCreator::create_laplace_matrix(
      dh, quadrature_formula, stiffness_matrix, {}, constraints);
    MatrixCreator::create_mass_matrix(
      dh, quadrature_formula, mass_matrix, {}, constraints);

    stiffness_inverse.initialize(stiffness_matrix);
    mass_inverse.initialize(mass_matrix);
  }

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> mass_matrix;

  SparseDirectUMFPACK stiffness_inverse;
  SparseDirectUMFPACK mass_inverse;
};

#endif // TestBench_H