#include "poisson.h"

#include <deal.II/numerics/error_estimator.h>

template <int dim>
PoissonParameters<dim>::PoissonParameters(const std::string &filename)
{
  prm.enter_subsection("Poisson parameters");
  {
    prm.add_parameter("Finite element degree", fe_degree);
    prm.add_parameter("Initial refinement", initial_refinement);
    prm.add_parameter("Number of cycles", n_cycles);
    prm.add_parameter("Exact solution expression", exact_solution_expression);
    prm.add_parameter("Right hand side expression", rhs_expression);
    prm.add_parameter("Grid name", grid_name);
    prm.add_parameter("Grid arguments", grid_arguments);
    prm.add_parameter("Output file name", output_file_name);
    prm.add_parameter("Output directory", output_directory);
  }
  prm.leave_subsection();

  prm.enter_subsection("Convergence table");
  convergence_table.add_parameters(prm);
  prm.leave_subsection();

  prm.enter_subsection("Local refinement");
  {
    prm.add_parameter("Refinement fraction", ref_frac);
    prm.add_parameter("Coarsening fraction", coarse_frac);
  }
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
  rhs_function.initialize(FunctionParser<dim>::default_variable_names(),
                          {rhs_expression},
                          constants);
}

//Definitions for Poisson
template <int dim>
Poisson<dim>::Poisson(const PoissonParameters<dim> &par)
  : par(par)
  , fe(par.fe_degree)
  , dof_handler(triangulation)
{}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::generate_from_name_and_arguments(triangulation,
                                                      par.grid_name,
                                                      par.grid_arguments);
  triangulation.refine_global(par.initial_refinement);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
Poisson<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           par.exact_solution,
                                           constraints);

  // Create hanging node constraints
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(2*fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            par.rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}



template <int dim>
void
Poisson<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}



template <int dim>
void
Poisson<dim>::output_results(const unsigned int cycle) const
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
Poisson<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        {
          Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
          KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim-1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);
    
          GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                      estimated_error_per_cell,
                                                      par.ref_frac, // Refinement fraction from parameter file
                                                     par.coarse_frac); // Coarsening fraction from parameter file
    
          triangulation.execute_coarsening_and_refinement();    
          //triangulation.refine_global(1);
        }
      setup_system();
      assemble_system();
      solve();
      output_results(cycle);
      par.convergence_table.error_from_exact(dof_handler,
                                             solution,
                                             par.exact_solution);
    }
  par.convergence_table.output_table(std::cout);
}

template class Poisson<1>;
template class Poisson<2>; 
template class Poisson<3>;
template struct PoissonParameters<1>;
template struct PoissonParameters<2>;
template struct PoissonParameters<3>;