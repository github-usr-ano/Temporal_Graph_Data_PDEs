#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>

#include <deal.II/base/hdf5.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>

#include <sstream>
#include <string>

#include <deal.II/grid/grid_out.h>
#include <math.h>

namespace Step23 {
using namespace dealii;

template <int dim> class WaveEquation {
public:
  WaveEquation();
  void run();

private:
  void setup_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> old_solution;
  Vector<double> system_rhs;

  double time_step;
  double time;
  unsigned int timestep_number;
  const double theta;
};

// Initial Values. This function is space dependent (p for point)
template <int dim> class InitialValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    // Define a small rectanble with positive initialization
    if ((p[0] > 3500000. && p[0] < 3600000.) &&
        (p[1] > 5400000. && p[1] < 5500000.)) {
      return 1;
    } else
      return 0;
  }
};

// This Function can make the diffusion-parameter time and space dependent.
// Since independent of the time component or the point "-1" is returned, this
// functionality is not reallyl used. For possible studies we anyways wrote this
// function
template <int dim> class Diffusion : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return -1.;
  }
};

// This function defines the source term S and is time and space dependent
// To generate the different sources, as described in the paper in the
// Supplementary Material A2, Table 3, we define in this function rectangles (in
// the paper called \Omega_S) by defining center_y, and center_x. we also define
// in this function \hat{s} which determines the intensity of the source term.
template <int dim> class RightHandSide : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    // For a better readability, we rescale the time, such that 80 timesteps are
    // producing to a timecomponent from  0 to 1
    float time_component = ((this->get_time() * 1e-10)); // from 0 to 1
    // modulo_time_component restarts from 0, after the experimental Id 10 from
    // Table 3 in A2.
    float modulo_time_component = fmod((time_component / 3), 6);
    float center_y;
    float center_x;
    // First 9 * 80 steps, this is the rectangles center
    if (modulo_time_component <= 2) {
      center_y = 3.45e6;
      center_x = 5.4e6;
    } else if ((modulo_time_component > 2) && (modulo_time_component <= 4)) {
      center_y = 3.6e6;
      center_x = 5.6e6;
    } else if ((modulo_time_component > 4) && (modulo_time_component <= 6)) {
      center_y = 3.45e6;
      center_x = 5.8e6;
    }

    // The intensity of the source term varies
    float mod_time_component = fmod(time_component, 6);
    if ((p[0] > center_y && p[0] < center_y * 1.01) &&
        (p[1] > center_x && p[1] < center_x * 1.01)) {
      if (mod_time_component < 0.1) {
        return -32 * 1e-10;
      } else if ((mod_time_component > 0.8) && (mod_time_component < 0.9)) {
        return -22 * 1e-10;
      } else if ((mod_time_component > 1.3) && (mod_time_component < 1.4)) {
        return -27 * 1e-10;
      } else if ((mod_time_component > 2.1) && (mod_time_component < 2.2)) {
        return -32 * 1e-10;
      } else if ((mod_time_component > 2.8) && (mod_time_component < 2.9)) {
        return -42 * 1e-10;
      } else if ((mod_time_component > 3.3) && (mod_time_component < 3.4)) {
        return -28 * 1e-10;
      } else if ((mod_time_component > 4.1) && (mod_time_component < 4.2)) {
        return -38 * 1e-10;
      } else if ((mod_time_component > 4.8) && (mod_time_component < 4.9)) {
        return -33 * 1e-10;
      } else if ((mod_time_component > 5.3) && (mod_time_component < 5.4)) {
        return -30 * 1e-10;
      } else
        return 0;
    } else
      return 0;
  }
};

// This function defines the Advection Field, in Table 3, A2 of the Aper, this
// is denotes as \beta Its interpretation can be understood as "wind speed"
template <int dim> class AdvectionField : public TensorFunction<1, dim> {
public:
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
};

template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const {
  Tensor<1, dim> value;
  // We rescaled the time for better readability
  float time_component = (this->get_time() * 1e-10) ;
  // The first 18 Experiment-IDs are created with this vector field
  if ((time_component / 18) <= 1) {
    value[0] = 0.00005;
    value[1] = -0.00005;
  }
  // The second 18 Experiment-IDs are created with this vector field
  else if (((time_component / 18) > 1) && ((time_component / 18) <= 2)) {
    value[0] = -0.00002 * ((p[1] * 1e-6 * 5) - 27.5);
    value[1] = 0.00006;
  }
  // The third 18 Experiment-IDs are created with this vector field
  else if (((time_component / 18) > 2) && ((time_component / 18) <= 3)) {
    value[0] = 0.00005 * ((p[1] * 1e-6 * 5) - 27.5);
    value[1] = 0.00005;
  }
  return value;
}

// This Function defines Dirichlet Boundary Values with value 0
template <int dim> class BoundaryValues : public Function<dim> {
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0.;
  }
};

template <int dim>
WaveEquation<dim>::WaveEquation()
    : fe(1), dof_handler(triangulation), time_step(1000000000. / 8),
      time(time_step), timestep_number(1), theta(0.5) {}

template <int dim> void WaveEquation<dim>::setup_system() {
  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f("Germany.msh"); // The mesh of the domain is loaded
  gridin.read_msh(f);

  std::ofstream out("germany_grid.svg");
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

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.close();
}

// Linear solver
template <int dim> void WaveEquation<dim>::solve() {
  SparseDirectUMFPACK A_direct;
  A_direct.solve(system_matrix, system_rhs);
  solution = system_rhs;
}
// Saving the results as VTK and csv files
template <int dim> void WaveEquation<dim>::output_results() const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(old_solution, "U");

  data_out.build_patches();

  const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 4) + ".vtk";
  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);
  std::ofstream output(filename);
  data_out.write_vtk(output);

  Vector<double> point_value(1);
  const std::string filename_csv = "solution-" + std::to_string(1) + "_" +
                                  Utilities::int_to_string(timestep_number, 4) +
                                  ".csv";
  std::ofstream csvfile;
  csvfile.open(filename_csv);

  csvfile << "Value \n";

  std::vector<std::vector<double>> sample_points_coords;

  // Read the CSV file containing the coordinates of the evaluation points,
  // extract the X and Y coordinates and store them in the
  // "sample_points_coords" vector
  std::ifstream file("germany_centers.csv");

  if (file.is_open()) {
    std::string line;
    std::getline(file, line); // Read the header line and ignore it

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string token;
      std::vector<double> coordinates;

      for (int i = 0; i < 3; ++i) {
        std::getline(iss, token, ',');
        if (i >= 1) // Extract X and Y coordinates from 3rd and 4th columns
        {
          coordinates.push_back(std::stod(token));
        }
      }
      sample_points_coords.push_back(coordinates);
    }
    file.close();
  } else {
    std::cerr << "Unable to open file germany_centers.csv" << std::endl;
  }

  // Evaluation of the solution at the given points
  // The value of the evaluation thes saved in a csvfile
  for (int i = 0; i < sample_points_coords.size(); i++) {
    try {
      VectorTools::point_value(
          dof_handler, old_solution,
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
  csvfile.close();
}

template <int dim> void WaveEquation<dim>::run() {
  setup_system();

  VectorTools::project(dof_handler, constraints, QGauss<dim>(fe.degree + 1),
                       InitialValues<dim>(), old_solution);
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  RightHandSide<dim> rhs_function;
  RightHandSide<dim> old_rhs_function;
  Diffusion<dim> diffusion_alpha;
  AdvectionField<dim> advection_field;
  AdvectionField<dim> old_advection_field;

  for (; time <= 54 * 1e10; time += time_step, ++timestep_number) // 54 = 3*6*3
  {
    rhs_function.set_time(time);
    old_rhs_function.set_time(time + time_step);
    advection_field.set_time(time);
    old_advection_field.set_time(time + time_step);
    system_matrix = 0;
    system_rhs = 0;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      std::vector<Tensor<1, dim>> old_solution_gradients(
          fe_values.n_quadrature_points);
      fe_values.get_function_gradients(old_solution, old_solution_gradients);

      std::vector<double> old_solution_values(fe_values.n_quadrature_points);
      fe_values.get_function_values(old_solution, old_solution_values);

      cell_matrix = 0;
      cell_rhs = 0;
      // system_rhs.add(theta * time_step, forcing_terms);
      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
        const auto &x_q = fe_values.quadrature_point(q_index);
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices()) {
            cell_matrix(i, j) +=
                ((fe_values.shape_value(i, q_index) *
                  fe_values.shape_value(j, q_index)) -
                 (time_step * (1 - theta) * fe_values.shape_value(i, q_index) *
                  (advection_field.value(x_q) *
                   fe_values.shape_grad(j, q_index))) -
                 (time_step * (1 - theta) * diffusion_alpha.value(x_q) *
                  fe_values.shape_grad(i, q_index) *
                  fe_values.shape_grad(j, q_index))) *
                fe_values.JxW(q_index);
          }
        for (const unsigned int i : fe_values.dof_indices()) {

          cell_rhs(i) +=
              ((fe_values.shape_value(i, q_index) *
                old_solution_values[q_index]) +
               (time_step * theta * fe_values.shape_value(i, q_index) *
                (old_advection_field.value(x_q) *
                 old_solution_gradients[q_index])) +
               (time_step * diffusion_alpha.value(x_q) * theta *
                fe_values.shape_grad(i, q_index) *
                old_solution_gradients[q_index]) -
               (time_step * theta * fe_values.shape_value(i, q_index) *
                rhs_function.value(x_q)) -
               (time_step * (1 - theta) * fe_values.shape_value(i, q_index) *
                old_rhs_function.value(x_q))) *
              fe_values.JxW(q_index);
        }
      }
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                            cell_matrix(i, j));
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
    std::map<types::global_dof_index, double> boundary_values;
    BoundaryValues<dim> boundary_values_function;
    boundary_values_function.set_time(time);

    VectorTools::interpolate_boundary_values(
        dof_handler, 0, boundary_values_function, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                       system_rhs);

    solve();
    std::cout << "Solved at time t=" << time << std::endl;
    output_results();
    old_solution = solution;
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
