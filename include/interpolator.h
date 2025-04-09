#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <string>

using namespace dealii;

template <int dim>
struct InterpolatorParameters
{
  InterpolatorParameters(const std::string &filename = "interpolator.prm");

  unsigned int fe_degree                 = 1;
  unsigned int initial_refinement        = 1;
  unsigned int n_cycles                  = 1;
  std::string  exact_solution_expression = "sin(2*pi*x)*sin(2*pi*y)";
  std::string  grid_name                 = "hyper_cube";
  std::string  grid_arguments            = "0 : 1 : false";
  std::string  output_file_name          = "solution";
  std::string  output_directory          = ".";

  FunctionParser<dim>            exact_solution;
  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};

template <int dim>
class Interpolator
{
public:
  Interpolator(const InterpolatorParameters<dim> &parameters);
  void
  run();

private:
  void
  make_grid();
  void
  setup_system();
  void
  interpolate();
  void
  output_results(const unsigned int cycle) const;

  const InterpolatorParameters<dim> &par;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  Vector<double> solution;
};

#endif // INTERPOLATOR_H