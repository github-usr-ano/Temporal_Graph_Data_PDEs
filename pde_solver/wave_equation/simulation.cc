#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>

#include <deal.II/base/hdf5.h>

#include <sstream>
#include <string>

#include <deal.II/grid/grid_out.h>

namespace Step23 {
using namespace dealii;

template <int dim> class WaveEquation {
public:
  WaveEquation();
  void run();

private:
  void setup_system();
  void solve_u();
  void solve_v();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> matrix_u;
  SparseMatrix<double> matrix_v;

  Vector<double> solution_u, solution_v;
  Vector<double> old_solution_u, old_solution_v;
  Vector<double> system_rhs;

  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;

  double time_step;
  double time;
  unsigned int timestep_number;
  const double theta;
  const double damping_b;
  const double dirichlet_alpha;
  const double neumann_beta;
};

// Initial Values of u
template <int dim> class InitialValuesU : public Function<dim> {
public:
  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

// Initial Values of v
template <int dim> class InitialValuesV : public Function<dim> {
public:
  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

// Legacy code kept to have a better comparison to tutorial step-23
// Additionally making it much easier to implement extensions to the wave
// equation
template <int dim> class RightHandSide : public Function<dim> {
public:
  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

// Dirichlet Boundary Values of U
// This function is time and space dependent
template <int dim> class BoundaryValuesU : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    // First wave startes on the left-upper corner before timestep 5000
    if ((this->get_time() <= 50000.) && (p[1] > 6200000.) && (p[0] < 3400000.))
      return 0.8;
    // Second wave startes on the right-upper corner between  time 1000000 and
    // 1100000.
    else if ((this->get_time() >= 1000000.) && (this->get_time() <= 1100000.) &&
             (p[1] > 6200000.) && (p[0] > 3800000.))
      return 1;
    // Third wave startes on the upper center corner between  time 1800000 and
    // 1900000.
    else if ((this->get_time() >= 1800000.) && (this->get_time() <= 1900000.) &&
             (p[0] > 3500000.) && (p[0] < 3600000.) && (p[1] > 6200000.))
      return 0.9;
    // Otherwise Dirichlet boundaries are zero
    else
      return 0;
  }
};

// Dirichlet Boundary Values of v
// This function is time and space dependent
template <int dim> class BoundaryValuesV : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

template <int dim>
WaveEquation<dim>::WaveEquation()
    : fe(1), dof_handler(triangulation), time_step(100000. / 64),
      time(time_step), timestep_number(1), theta(0.5), damping_b(0.000005),
      dirichlet_alpha(0.5), neumann_beta(0.5) {}

template <int dim> void WaveEquation<dim>::setup_system() {
  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f("Germany_coastline.msh"); // The mesh of the domain is loaded
  gridin.read_msh(f);

  triangulation.refine_global(1);

  std::ofstream out("grid-1.svg");
  GridOut grid_out;
  grid_out.write_svg(triangulation, out);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  matrix_u.reinit(sparsity_pattern);
  matrix_v.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1),
                                       laplace_matrix);

  solution_u.reinit(dof_handler.n_dofs());
  solution_v.reinit(dof_handler.n_dofs());
  old_solution_u.reinit(dof_handler.n_dofs());
  old_solution_v.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  cell_rhs.reinit(fe.n_dofs_per_cell());
  constraints.close();
}

// This function solves the linear system to receive u
// Note that the solver is a CG (Conjugated Gradient) solver.
template <int dim> void WaveEquation<dim>::solve_u() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

  std::cout << "   u-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

// This function solves the linear system to receive v
// Note that the solver is a CG (Conjugated Gradient) solver.
template <int dim> void WaveEquation<dim>::solve_v() {
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

  std::cout << "   v-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

// Output the solution U as VTK and CSV files
template <int dim> void WaveEquation<dim>::output_results() const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "U");

  data_out.build_patches();

  const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 5) + ".vtk";
  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);
  std::ofstream output(filename);
  data_out.write_vtk(output);

  Vector<double> point_value(1);
  std::vector<std::vector<double>> sample_points_coords;

  const std::string filename_csv =
      "solution-" + Utilities::int_to_string(timestep_number, 5) + ".csv";
  std::ofstream csvfile;
  csvfile.open(filename_csv);
  csvfile << "Wave-height \n";

  // Read the CSV file containing the coordinates of the evaluation points,
  // extract the X and Y coordinates
  // and store them in the "sample_points_coords" vector
  std::ifstream file("Germany_coastline_points.csv");

  if (file.is_open()) {
    std::string line;
    std::getline(file, line); // Read the header line and ignore it
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string token;
      std::vector<double> coordinates;
      // x, y coordinates
      for (int i = 0; i < 3; ++i) {
        std::getline(iss, token, ',');
        coordinates.push_back(std::stod(token));
      }
      sample_points_coords.push_back(coordinates);
    }
    file.close();
  } else {
    std::cerr << "Unable to open file Germany_coastline_points.csv"
              << std::endl;
  }

  // Evaluation of the solution at the given points
  // The value of the evaluation thes saved in a csvfile
  for (int i = 0; i < sample_points_coords.size(); i++) {
    try {
      VectorTools::point_value(
          dof_handler, solution_u,
          Point<2>(sample_points_coords[i][0], sample_points_coords[i][1]),
          point_value);
      csvfile << point_value[0] << ""
              << "\n";
    } catch (...) {
      std::cout << "Not working on " << i << std::endl;
      std::cout << "Points there are:  " << sample_points_coords[i][0]
                << "and: " << sample_points_coords[i][1] << std::endl;
    }
  }
}

template <int dim> void WaveEquation<dim>::run() {
  setup_system();

  VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                       InitialValuesU<dim>(), old_solution_u);
  VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                       InitialValuesV<dim>(), old_solution_v);

  Vector<double> tmp(solution_u.size());
  Vector<double> forcing_terms(solution_u.size());

  // Loop over the time. at time t = 2700000 the simulation stops
  for (; time <= 2700000; time += time_step, ++timestep_number) {
    std::cout << "Time step " << timestep_number << " at t=" << time
              << std::endl;

    // Set-up of the equation to solve for u.
    // for more information we refer to the documentation of deal.ii and
    // tutorial step-23
    mass_matrix.vmult(system_rhs, old_solution_u);
    system_rhs *= (1 + time_step * theta * damping_b);

    mass_matrix.vmult(tmp, old_solution_v);
    system_rhs.add(time_step, tmp);

    laplace_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

    RightHandSide<dim> rhs_function;
    rhs_function.set_time(time);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1),
                                        rhs_function, tmp);

    forcing_terms = tmp;
    forcing_terms *= theta * time_step;

    rhs_function.set_time(time - time_step);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1),
                                        rhs_function, tmp);

    forcing_terms.add((1 - theta) * time_step, tmp);

    system_rhs.add(theta * time_step, forcing_terms);

    matrix_u.copy_from(mass_matrix);
    matrix_u *= (1 + time_step * theta * damping_b);
    matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);

    FEFaceValues<dim> fe_face_values(fe, QGauss<dim - 1>(fe.degree + 1),
                                     update_values | update_JxW_values);

    // This line decides if the Robin or Dirichlet conditions are applied
    // If decider is set to true, Dirichlet conditions are applied,
    // If decider is set to false, Robin conditions are applied
    bool decider =
        ((time <= 70000.) || ((time >= 1000000.) && (time <= 1120000.)) ||
         (((time >= 1800000.) && (time <= 1930000.))));

    // Robin Boundary conditions
    if (!decider) {
      std::vector<double> old_solution_values_u(
          QGauss<dim - 1>(fe.degree + 1).size());
      for (const auto &cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary()) {
            fe_face_values.reinit(cell, face);
            fe_face_values.get_function_values(old_solution_u,
                                               old_solution_values_u);
            for (const unsigned int q_index :
                 fe_face_values.quadrature_point_indices()) {
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j) {
                  cell_matrix(i, j) += time_step * time_step * theta * theta *
                                       (1 / neumann_beta) * dirichlet_alpha *
                                       fe_face_values.shape_value(i, q_index) *
                                       fe_face_values.shape_value(j, q_index) *
                                       fe_face_values.JxW(q_index);
                }
                cell_rhs(i) += -time_step * time_step * theta * (1 - theta) *
                               (1 / neumann_beta) * dirichlet_alpha *
                               fe_face_values.shape_value(i, q_index) *
                               old_solution_values_u[q_index] *
                               fe_face_values.JxW(q_index);
              }
            }
            std::vector<types::global_dof_index> local_dof_indices(
                fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_face_values.dof_indices()) {
              for (const unsigned int j : fe_face_values.dof_indices())
                matrix_u.add(local_dof_indices[i], local_dof_indices[j],
                             cell_matrix(i, j));
              system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
          }
      }
    }
    // Dirichlet Boundary conditions
    else {
      BoundaryValuesU<dim> boundary_values_u_function;
      boundary_values_u_function.set_time(time);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, boundary_values_u_function, boundary_values);

      MatrixTools::apply_boundary_values(boundary_values, matrix_u, solution_u,
                                         system_rhs);
    }

    solve_u(); // Solve the linear system for u

    // Set-up of the equation to solve for v.
    // for more information we refer to the documentation of deal.ii and
    // tutorial step-23
    laplace_matrix.vmult(system_rhs, solution_u);
    system_rhs *= -theta * time_step;

    mass_matrix.vmult(tmp, old_solution_v);
    system_rhs += tmp;

    mass_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(damping_b, tmp);

    mass_matrix.vmult(tmp, solution_u);
    system_rhs.add(-damping_b, tmp);

    laplace_matrix.vmult(tmp, old_solution_u);
    system_rhs.add(-time_step * (1 - theta), tmp);

    system_rhs += forcing_terms;

    matrix_v.copy_from(mass_matrix);

    // Robin Boundary conditions
    if (!decider) {
      std::vector<double> solution_values_u(
          QGauss<dim - 1>(fe.degree + 1).size());
      std::vector<double> old_solution_values_u(
          QGauss<dim - 1>(fe.degree + 1).size());
      for (const auto &cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary()) {
            fe_face_values.reinit(cell, face);
            fe_face_values.get_function_values(old_solution_u,
                                               old_solution_values_u);
            fe_face_values.get_function_values(solution_u, solution_values_u);
            for (const unsigned int q_index :
                 fe_face_values.quadrature_point_indices()) {
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
                cell_rhs(i) += -time_step * theta * (1 / neumann_beta) *
                                   dirichlet_alpha *
                                   fe_face_values.shape_value(i, q_index) *
                                   solution_values_u[q_index] *
                                   fe_face_values.JxW(q_index) -
                               time_step * (1 - theta) * (1 / neumann_beta) *
                                   dirichlet_alpha *
                                   fe_face_values.shape_value(i, q_index) *
                                   old_solution_values_u[q_index] *
                                   fe_face_values.JxW(q_index);
              }
            }
            std::vector<types::global_dof_index> local_dof_indices(
                fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_face_values.dof_indices()) {
              system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
          }
      }
    }
    // Dirichlet Boundary conditions
    else {
      BoundaryValuesV<dim> boundary_values_v_function;
      boundary_values_v_function.set_time(time);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, boundary_values_v_function, boundary_values);
      MatrixTools::apply_boundary_values(boundary_values, matrix_v, solution_v,
                                         system_rhs);
    }
    solve_v(); // Solve the linear system for v

    output_results();

    std::cout << "   Total energy: "
              << (mass_matrix.matrix_norm_square(solution_v) +
                  laplace_matrix.matrix_norm_square(solution_u)) /
                     2
              << std::endl;

    old_solution_u = solution_u;
    old_solution_v = solution_v;
  }
}
} // namespace Step23

int main(int argc, char *argv[]) {
  try {
    using namespace Step23;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    WaveEquation<2> wave_equation_solver;
    wave_equation_solver.run();
  } catch (std::exception &exc) {
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
  } catch (...) {
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
