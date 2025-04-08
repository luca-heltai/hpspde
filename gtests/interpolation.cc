#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>


using namespace dealii;

TEST(Interpolation, CheckInterpolation)
{
  static const int   dim = 2;
  Triangulation<dim> triangulation;

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  FE_Q<dim> fe(1);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());

  FunctionParser<dim>           fun;
  std::map<std::string, double> constants;
  fun.initialize("x,y", "sin(x*y)", constants);


  VectorTools::interpolate(dof_handler, fun, solution);
  Vector<double> cell_difference(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    fun,
                                    cell_difference,
                                    QGauss<dim>(3),
                                    VectorTools::L2_norm);
  double L2_error = cell_difference.l2_norm();

  EXPECT_LT(L2_error, 1e-1);
}