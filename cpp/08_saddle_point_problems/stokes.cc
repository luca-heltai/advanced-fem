/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symbolic_function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;



template <int dim>
class Step6
{
public:
  Step6();

  void
  run();

private:
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  refine_grid();
  void
  output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  FE_Q<dim>     fe_velocity;
  FE_DGP<dim>   fe_pressure;
  FESystem<dim> fe;

  FEValuesExtractors::Vector velocity;
  FEValuesExtractors::Scalar pressure;

  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;
};



template <int dim>
Step6<dim>::Step6()
  : fe_velocity(2)
  , fe_pressure(1)
  , fe(fe_velocity, dim, fe_pressure, 1)
  , velocity(0)
  , pressure(dim)
  , dof_handler(triangulation)
{}



template <int dim>
void
Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);


  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(dim +
                                                                        1),
                                           constraints,
                                           fe.component_mask(velocity));

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void
Step6<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Functions::SymbolicFunction<dim> rhs_function(
    dim == 2 ?
      "x^2*y^2*(x - 1)^2*(2*y - 2) + 2*x^2*y*(x - 1)^2*(y - 1)^2;-x^2*y^2*(2*x - 2)*(y - 1)^2 - 2*x*y^2*(x - 1)^2*(y - 1)^2; 0" :
      "x^2*y^2*(x - 1)^2*(2*y - 2) + 2*x^2*y*(x - 1)^2*(y - 1)^2;-x^2*y^2*(2*x - 2)*(y - 1)^2 - 2*x*y^2*(x - 1)^2*(y - 1)^2; 0; 0");

  Vector<double> rhs_values(dim + 1);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          rhs_function.vector_value(fe_values.quadrature_point(q_index),
                                    rhs_values);
          for (const unsigned int i : fe_values.dof_indices())
            {
              auto v      = fe_values[velocity].value(i, q_index);
              auto div_v  = fe_values[velocity].divergence(i, q_index);
              auto grad_v = fe_values[velocity].gradient(i, q_index);
              auto q      = fe_values[pressure].value(i, q_index);

              for (const unsigned int j : fe_values.dof_indices())
                {
                  auto div_u  = fe_values[velocity].divergence(j, q_index);
                  auto grad_u = fe_values[velocity].gradient(j, q_index);
                  auto p      = fe_values[pressure].value(j, q_index);

                  cell_matrix(i, j) +=
                    (scalar_product(grad_u, grad_v) - div_v * p - div_u * q) *
                    fe_values.JxW(q_index); // dx
                }
              for (unsigned int d = 0; d < dim; ++d)
                cell_rhs(i) += (1000 * rhs_values[d] *   // f(x)
                                v[d] *                   // phi_i(x_q)
                                fe_values.JxW(q_index)); // dx
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}



template <int dim>
void
Step6<dim>::solve()
{
  SparseDirectUMFPACK inverse;
  inverse.initialize(system_matrix);

  // SolverControl            solver_control(1000, 1e-12);
  // SolverCG<Vector<double>> solver(solver_control);

  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(system_matrix, 1.2);

  // solver.solve(system_matrix, solution, system_rhs, preconditioner);

  inverse.vmult(solution, system_rhs);
  constraints.distribute(solution);
}



template <int dim>
void
Step6<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell,
                                     fe.component_mask(velocity));

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  triangulation.execute_coarsening_and_refinement();
}



template <int dim>
void
Step6<dim>::output_results(const unsigned int cycle) const
{
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches(fe.degree);

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
  }
}



template <int dim>
void
Step6<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation);
          triangulation.refine_global(4);
        }
      else
        refine_grid();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;

      setup_system();

      std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
    }
}



int
main()
{
  try
    {
      Step6<2> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
