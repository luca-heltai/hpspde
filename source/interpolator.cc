#include "interpolator.h"

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// Definitions for InterpolatorParameters
template <int dim>
InterpolatorParameters<dim>::InterpolatorParameters(const std::string &filename)
{
  prm.enter_subsection("Interpolator parameters");
  {
    prm.add_parameter("Finite element degree", fe_degree);
    prm.add_parameter("Initial refinement", initial_refinement);
    prm.add_parameter("Number of cycle", n_cycles);
    prm.add_parameter("Exact solution expression", exact_solution_expression);
    prm.add_parameter("Grid name", grid_name);
    prm.add_parameter("Grid arguments", grid_arguments);
    prm.add_parameter("Output file name", output_file_name);
    prm.add_parameter("Output directory", output_directory);
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

// Definitions for Interpolator
template <int dim>
Interpolator<dim>::Interpolator(const InterpolatorParameters<dim> &par)
  : par(par)
  , fe(par.fe_degree)
  , dof_handler(triangulation)
{}

// ...existing code...
template <int dim>
void
Interpolator<dim>::make_grid()
{
  GridGenerator::generate_from_name_and_arguments(triangulation,
                                                  par.grid_name,
                                                  par.grid_arguments);
  triangulation.refine_global(par.initial_refinement);
}


template <int dim>
void
Interpolator<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
}



template <int dim>
void
Interpolator<dim>::interpolate()
{
  VectorTools::interpolate(dof_handler, par.exact_solution, solution);
  par.convergence_table.error_from_exact(dof_handler,
                                         solution,
                                         par.exact_solution);
}


template <int dim>
void
Interpolator<dim>::output_results(const unsigned int cycle) const
{
  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.write_higher_order_cells = true;
  DataOut<dim> data_out;
  data_out.set_flags(vtk_flags);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();
  std::string fname = par.output_file_name + "-" + std::to_string(dim) +
                      "d-cycle-" + std::to_string(cycle);
  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back(std::make_pair(cycle, fname + ".vtu"));

  std::ofstream vtuoutput(par.output_directory + "/" + fname + ".vtu");
  std::ofstream pvdoutput(par.output_directory + "/" + fname + ".pvd");
  data_out.write_vtu(vtuoutput);
  DataOutBase::write_pvd_record(pvdoutput, times_and_names);
}


template <int dim>
void
Interpolator<dim>::run()
{
  std::cout << "Interpolating solution in " << dim << "D" << std::endl;
  std::cout << "   Finite element degree: " << par.fe_degree << std::endl;
  std::cout << "   Initial refinement: " << par.initial_refinement << std::endl;
  std::cout << "   Number of cycles: " << par.n_cycles << std::endl;
  std::cout << "   Exact solution expression: " << par.exact_solution_expression
            << std::endl;

  for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        triangulation.refine_global(1);
      setup_system();
      interpolate();
      output_results(cycle);
    }
  par.convergence_table.output_table(std::cout);
}

template class Interpolator<1>;
template class Interpolator<2>;
template class Interpolator<3>;
template struct InterpolatorParameters<1>;
template struct InterpolatorParameters<2>;
template struct InterpolatorParameters<3>;