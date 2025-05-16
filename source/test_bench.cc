#include "test_bench.h"

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// Definitions for TestBenchParameters
template <int dim, int spacedim>
TestBenchParameters<dim, spacedim>::TestBenchParameters(
  const std::string &filename)
{
  prm.enter_subsection("TestBench parameters");
  {
    prm.add_parameter("Finite element name", fe_name);
    prm.add_parameter("Initial refinement", initial_refinement);
    prm.add_parameter("Exact solution expression", exact_solution_expression);
    prm.add_parameter("Grid name", grid_name);
    prm.add_parameter("Grid arguments", grid_arguments);
    prm.add_parameter("Output file name", output_file_name);
    prm.add_parameter("Output directory", output_directory);
    prm.add_parameter("DoF renumbering", renumbering);
    prm.add_parameter("omega", omega);
    prm.add_parameter("Smoothing steps", smoothing_steps);
  }
  prm.leave_subsection();

  prm.enter_subsection("Convergence table");
  convergence_table.add_parameters(prm);
  prm.leave_subsection();

  try
    {
      prm.parse_input(filename);
    }
  catch (std::exception &exc)
    {
      prm.print_parameters(filename, ParameterHandler::Short);
      prm.parse_input(filename);
    }
  std::map<std::string, double> constants;
  constants["pi"] = numbers::PI;
  exact_solution.initialize(FunctionParser<dim>::default_variable_names(),
                            {exact_solution_expression},
                            constants);
}

// Definitions for TestBench
template <int dim, int spacedim>
TestBench<dim, spacedim>::TestBench(
  const TestBenchParameters<dim, spacedim> &par)
  : par(par)
  , dof_handler(triangulation)
{}

template <int dim, int spacedim>
void
TestBench<dim, spacedim>::make_grid()
{
  GridGenerator::generate_from_name_and_arguments(triangulation,
                                                  par.grid_name,
                                                  par.grid_arguments);
  triangulation.refine_global(par.initial_refinement);
}


template <int dim, int spacedim>
void
TestBench<dim, spacedim>::setup_system()
{
  fe = FETools::get_fe_by_name<dim, spacedim>(par.fe_name);
  dof_handler.distribute_dofs(*fe);
  for (const auto &renumber : par.renumbering)
    {
      if (renumber == "Cuthill_McKee")
        DoFRenumbering::Cuthill_McKee(dof_handler);
      else if (renumber == "component_wise")
        DoFRenumbering::component_wise(dof_handler);
      else if (renumber == "support_point_wise")
        DoFRenumbering::support_point_wise(dof_handler);
      else if (renumber == "block_wise")
        DoFRenumbering::block_wise(dof_handler);
      else if (renumber == "hierarchical")
        DoFRenumbering::hierarchical(dof_handler);
      else if (renumber == "lexicographic")
        {
          if constexpr (dim > 1 && dim == spacedim)
            {
              BoundingBox<spacedim> bounding_box(triangulation.get_vertices());
              const auto max = bounding_box.get_boundary_points().second.norm();
              Tensor<1, spacedim> direction;
              for (unsigned int i = 0; i < spacedim; ++i)
                direction[spacedim - i - 1] = (std::pow(10u, i)) * max;

              DoFRenumbering::downstream(dof_handler, direction, true);
            }
          else
            {
              AssertThrow(false,
                          ExcMessage("Not implemented for dim/spacedim: " +
                                     renumber));
            }
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("Unknown renumbering strategy: " + renumber));
        }
    }
}

template <int dim, int spacedim>
auto
TestBench<dim, spacedim>::make_vector() const -> Vector<double>
{
  return Vector<double>(dof_handler.n_dofs());
};



template <int dim, int spacedim>
void
TestBench<dim, spacedim>::interpolate()
{
  VectorTools::interpolate(dof_handler, par.exact_solution, solution);
  par.convergence_table.error_from_exact(dof_handler,
                                         solution,
                                         par.exact_solution);
}


template <int dim, int spacedim>
void
TestBench<dim, spacedim>::output_results(const Vector<double> &output_field,
                                         const std::string    &suffix) const
{
  DataOut<dim, spacedim> data_out;
  if constexpr (dim > 1)
    {
      DataOutBase::VtkFlags vtk_flags;
      vtk_flags.write_higher_order_cells = true;
      data_out.set_flags(vtk_flags);
    }

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(output_field, "solution");

  data_out.build_patches();
  std::string fname =
    par.output_file_name + (suffix == "" ? "" : "_" + suffix) + ".vtk";

  std::ofstream vtkoutput(fname);
  data_out.write_vtk(vtkoutput);
}


template <int dim, int spacedim>
void
TestBench<dim, spacedim>::initialize()
{
  make_grid();
  setup_system();
}


// Explicit instantiations
template struct TestBenchParameters<1, 1>;
template struct TestBenchParameters<1, 2>;
template struct TestBenchParameters<1, 3>;
template struct TestBenchParameters<2, 2>;
template struct TestBenchParameters<2, 3>;
template struct TestBenchParameters<3, 3>;

template class TestBench<1, 1>;
template class TestBench<1, 2>;
template class TestBench<1, 3>;
template class TestBench<2, 2>;
template class TestBench<2, 3>;
template class TestBench<3, 3>;