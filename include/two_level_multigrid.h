#ifndef TWO_LEVEL_MULTIGRID_H
#define TWO_LEVEL_MULTIGRID_H

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

using namespace dealii;

using LinOp = LinearOperator<Vector<double>>;

/**
 * Creates restriction and prolongation operators between two DoF handlers
 * using MGTwoLevelTransferNonNested.
 *
 * @param fine_dof_handler The fine level DoF handler
 * @param coarse_dof_handler The coarse level DoF handler
 * @return A pair of linear operators: (restriction, prolongation)
 *         restriction maps from fine to coarse
 *         prolongation maps from coarse to fine
 */
template <int dim>
auto
create_transfer_operators(const DoFHandler<dim> &fine_dof_handler,
                          const DoFHandler<dim> &coarse_dof_handler)
  -> std::pair<LinOp, LinOp>
{
  // Create the transfer object
  MGTwoLevelTransferNonNested<dim, Vector<double>> transfer;

  // Build the transfer operators
  transfer.reinit(fine_dof_handler, coarse_dof_handler);

  // Create restriction operator (fine -> coarse)
  LinOp restriction;
  restriction.reinit_range_vector = [&coarse_dof_handler](Vector<double> &v,
                                                          bool) {
    v.reinit(coarse_dof_handler.n_dofs());
  };
  restriction.reinit_domain_vector = [&fine_dof_handler](Vector<double> &v,
                                                         bool) {
    v.reinit(fine_dof_handler.n_dofs());
  };
  restriction.vmult = [&transfer](Vector<double>       &dst,
                                  const Vector<double> &src) {
    dst = 0; // Clear destination before restrict_and_add
    transfer.restrict_and_add(dst, src);
  };
  restriction.Tvmult = [&transfer](Vector<double>       &dst,
                                   const Vector<double> &src) {
    dst = 0; // Clear destination before prolongate_and_add
    transfer.prolongate_and_add(dst, src);
  };

  // Create prolongation operator (coarse -> fine)
  LinOp prolongation;
  prolongation.reinit_range_vector = [&fine_dof_handler](Vector<double> &v,
                                                         bool) {
    v.reinit(fine_dof_handler.n_dofs());
  };
  prolongation.reinit_domain_vector = [&coarse_dof_handler](Vector<double> &v,
                                                            bool) {
    v.reinit(coarse_dof_handler.n_dofs());
  };
  prolongation.vmult = [&transfer](Vector<double>       &dst,
                                   const Vector<double> &src) {
    dst = 0; // Clear destination before prolongate_and_add
    transfer.prolongate_and_add(dst, src);
  };
  prolongation.Tvmult = [&transfer](Vector<double>       &dst,
                                    const Vector<double> &src) {
    dst = 0; // Clear destination before restrict_and_add
    transfer.restrict_and_add(dst, src);
  };

  return std::make_pair(restriction, prolongation);
}

/**
 * Creates a two-level multigrid linear operator implementing a V-cycle.
 *
 * The V-cycle algorithm:
 * 1. Apply presmoother
 * 2. Restrict residual to coarse level
 * 3. Solve coarse problem exactly
 * 4. Prolongate correction to fine level
 * 5. Apply postsmoother
 *
 * @param fine_matrix The fine level system matrix
 * @param presmoother Pre-smoothing operator
 * @param restriction Restriction operator (fine to coarse)
 * @param coarse_solver Coarse level solver
 * @param prolongation Prolongation operator (coarse to fine)
 * @param postsmoother Post-smoothing operator
 * @return Linear operator representing the two-level multigrid preconditioner
 */
inline auto
two_level_multigrid(const LinOp &fine_matrix,
                    const LinOp &presmoother,
                    const LinOp &restriction,
                    const LinOp &coarse_solver,
                    const LinOp &prolongation,
                    const LinOp &postsmoother) -> LinOp
{
  // Create identity operator
  auto Id = identity_operator(fine_matrix);

  // V-cycle: x_new = x_old + correction
  // where correction comes from:
  // 1. Pre-smooth: x1 = x0 + S_pre * (b - A*x0)
  // 2. Coarse correct: x2 = x1 + P * A_coarse^(-1) * R * (b - A*x1)
  // 3. Post-smooth: x3 = x2 + S_post * (b - A*x2)

  // For a preconditioner P applied to residual r:
  // P*r = S_pre*r + P*A_coarse^(-1)*R*(I - A*S_pre)*r + S_post*(I - A*(S_pre +
  // P*A_coarse^(-1)*R*(I - A*S_pre)))*r

  // Coarse grid correction operator
  auto coarse_correction = prolongation * coarse_solver * restriction;

  // Step 1: Pre-smoothing contribution
  // Step 2: Coarse correction applied to residual after pre-smoothing
  auto residual_after_presmooth = Id - fine_matrix * presmoother;
  auto coarse_contrib           = coarse_correction * residual_after_presmooth;

  // Step 3: Combined correction so far
  auto combined_correction = presmoother + coarse_contrib;

  // Step 4: Post-smoothing applied to residual after combined correction
  auto residual_after_coarse = Id - fine_matrix * combined_correction;
  auto post_contrib          = postsmoother * residual_after_coarse;

  // Final V-cycle operator
  auto vcycle = combined_correction + post_contrib;

  return vcycle;
}

#endif // TWO_LEVEL_MULTIGRID_H