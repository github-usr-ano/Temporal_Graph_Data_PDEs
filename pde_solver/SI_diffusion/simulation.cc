#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/hdf5.h>

#include <stdexcept>

namespace Step23 {
using namespace dealii;

// Function that returns the diffusion coefficient D(x,t) from the paper (eq 1).
// This function is space dependent and easily can be advanced to also be
// time-independent
template <int dim>
double diffusion_coefficient(const Point<dim> &p, double &time,
                             int setting_param) {
  setting_param = ((setting_param / 25) % 5);
  switch (setting_param) {
  case 0:
    return 2e8;
  case 1:
    return (1 + (0.1 * std::sin(0.000005 * numbers::PI * p[0]) *
                 std::sin(0.000005 * numbers::PI * p[1]))) *
           1e8;
  case 2:
    return 3e8;
  case 3:
    return 8e7;
  case 4:
    return (1 + (0.2 * std::sin(0.000005 * numbers::PI * p[0]) *
                 std::sin(0.000005 * numbers::PI * p[1]))) *
           1e8;
  default:
    throw std::invalid_argument("Invalid setting_param");
  }
}

// Function that returns the r_value r(x,t)  from the paper (eq 1). This
// function is space dependent and easily can be advanced to also be
// time-independent
template <int dim>
double r_value(const Point<dim> &p, double &time, int exp_param) {
  exp_param = (exp_param / 5) % 5;
  switch (exp_param) {
  case 0:
    return 0.6; // constant
  case 1:
    return 1 +
           (0.2 * std::sin(0.000005 * numbers::PI * p[0]) *
            std::sin(0.000005 * numbers::PI * p[1])); // space dependent r_value
  case 2:
    return 0.7 + (0.6 * std::sin(0.000005 * numbers::PI * p[0]) *
                  std::sin(0.000005 * numbers::PI * p[1]));
  case 3:
    return 0.5 + (0.3 * std::sin(0.000002 * numbers::PI * p[0]) *
                  std::sin(0.000005 * numbers::PI * p[1]));
  case 4:
    return 1.1 + (0.1 * std::sin(0.000008 * numbers::PI * p[0]) *
                  std::sin(0.000005 * numbers::PI * p[1]));
  default:
    throw std::invalid_argument(
        "experiment identifier makes no sense while setting the r_value");
  }
}

template <int dim> class SIDiffusionEquation {
public:
  SIDiffusionEquation(int initial_param);
  void run();

private:
  void setup_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FESystem<dim> fe_system;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> old_euler_solution;
  Vector<double> old_euler_solution_add;
  Vector<double> system_rhs;
  Vector<double> euler_global_rhs;
  Vector<double> newton_update;

  double time_step;
  double time;
  unsigned int timestep_number;
  const double theta;
  const double newton_stepsize;
  const double newton_threshold;
  const double diffusion_d;
  const double transmission_r;
  const double alpha_duration;
  unsigned int initial_parameter;
};

// Initial Values for each run of of 364 time-steps. for both Susceptible and
// Infected
//
// We only run experiments with the first initialiization option (i.e.
// scen_zero, but wanted to keep this function for others to use.
//
// The InitialValuesSusceptible class will set in its constructor where the
// function_point will point to, depending on init_param_arg value
template <int dim> class InitialValuesSusceptible : public Function<dim> {
public:
  using function_pointer = double (*)(const Point<dim> &,
                                      unsigned int); // Function Pointer

  static double scen_zero(const Point<dim> &p,
                          const unsigned int component = 0) {
    if (component == 0) // Susceptible
    {
      return 1.;
    } else if (component == 1) // Infected
    {
      return 0.00;
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  static double scen_one(const Point<dim> &p,
                         const unsigned int component = 0) {
    if (component == 0) // Susceptible, = 1-infected
    {
      return 1 - (0.1 + (0.1 * std::sin(0.000005 * numbers::PI * p[0]) *
                         std::sin(0.000005 * numbers::PI * p[1])));
    } else if (component == 1) // Infected
    {
      return 0.1 + (0.1 * std::sin(0.000005 * numbers::PI * p[0]) *
                    std::sin(0.000005 * numbers::PI * p[1]));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  static double scen_two(const Point<dim> &p,
                         const unsigned int component = 0) {
    if (component == 0) // Susceptible, = 1-infected
    {
      return 1 - (0.15 - (0.15 * std::sin(0.000005 * numbers::PI * p[0]) *
                          std::sin(0.000005 * numbers::PI * p[1])));
    } else if (component == 1) // Infected
    {
      return 0.15 - (0.15 * std::sin(0.000005 * numbers::PI * p[0]) *
                     std::sin(0.000005 * numbers::PI * p[1]));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  static double scen_three(const Point<dim> &p,
                           const unsigned int component = 0) {
    if (component == 0) // Susceptible, = 1-infected
    {
      return 1 - (0.1 - (0.1 * std::sin(0.000004 * numbers::PI * p[0]) *
                         std::sin(0.000005 * numbers::PI * p[1])));
    } else if (component == 1) // Infected
    {
      return 0.1 - (0.1 * std::sin(0.000004 * numbers::PI * p[0]) *
                    std::sin(0.000005 * numbers::PI * p[1]));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  static double scen_four(const Point<dim> &p,
                          const unsigned int component = 0) {
    if (component == 0) // Susceptible, = 1-infected
    {
      return 1 - (0.1 - (0.1 * std::sin(0.000005 * numbers::PI * p[0]) *
                         std::sin(0.000002 * numbers::PI * p[1])));
    } else if (component == 1) // Infected
    {
      return 0.1 - (0.1 * std::sin(0.000005 * numbers::PI * p[0]) *
                    std::sin(0.000002 * numbers::PI * p[1]));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  // Constructer of the Initial Value class, which determines where the pointer
  // points to
  InitialValuesSusceptible(const unsigned int init_param_arg)
      : Function<dim>(2, 0) {
    int init_param = std::max(static_cast<int>(init_param_arg), 125) % 5;
    init_param = 0;
    switch (init_param) {
    case 0:
      option_function = &scen_zero;
      break;
    case 1:
      option_function = &scen_one;
      break;
    case 2:
      option_function = &scen_two;
      break;
    case 3:
      option_function = &scen_three;
      break;
    case 4:
      option_function = &scen_four;
      break;
    }
  }

  // Here only the value function gets called. It then calls function on which
  // the  option_function is pointing to
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    option_function(p, component);
  }

private:
  function_pointer option_function;
};

// Boundary condition for each run of of 364 time-steps. for both Susceptible
// and Infected works similar to the InitialValuesSusceptible class
//
// The BoundaryValuesSusceptible class will set in its constructor where the
// function_point will point to, depending on init_param_arg value
template <int dim> class BoundaryValuesSusceptible : public Function<dim> {
public:
  // Constructor
  // Constructer of the Initial Value class, which determines where the pointer
  // points to
  BoundaryValuesSusceptible(int init_param) : Function<dim>(2, 0) {
    init_param = (init_param / 125) % 5;
    switch (init_param) {
    case 0:
      option_function = &BoundaryValuesSusceptible::scen_zero;
      break;
    case 1:
      option_function = &BoundaryValuesSusceptible::scen_one;
      break;
    case 2:
      option_function = &BoundaryValuesSusceptible::scen_two;
      break;
    case 3:
      option_function = &BoundaryValuesSusceptible::scen_three;
      break;
    case 4:
      option_function = &BoundaryValuesSusceptible::scen_four;
      break;
    default:
      throw std::invalid_argument(
          "experiment identifier makes no sense in BoundaryValuesSusceptible");
    }
  }

  // Here only the value function gets called. It then calls function on which
  // the  option_function is pointing to
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component <= 2, ExcIndexRange(component, 0, 2));
    return (this->*option_function)(p, component);
  }

private:
  double (BoundaryValuesSusceptible::*option_function)(
      const Point<dim> &p, const unsigned int component) const;
  // Definition of the scenarios
  double scen_zero(const Point<dim> &p,
                   const unsigned int component = 0) const {
    if (component == 0) {
      return 1 -
             0.3 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else if (component == 1) {
      return 0.3 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  double scen_one(const Point<dim> &p, const unsigned int component = 0) const {
    if (component == 0) {
      return 1 -
             0.3 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else if (component == 1) {
      return 0.3 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  double scen_two(const Point<dim> &p, const unsigned int component = 0) const {
    if (component == 0) {
      return 1 -
             0.1 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else if (component == 1) {
      return 0.1 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  double scen_three(const Point<dim> &p,
                    const unsigned int component = 0) const {
    if (component == 0) {
      return 1 -
             0.5 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else if (component == 1) {
      return 0.5 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }

  double scen_four(const Point<dim> &p,
                   const unsigned int component = 0) const {
    if (component == 0) {
      return 1 -
             0.6 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else if (component == 1) {
      return 0.6 * std::abs(std::sin(this->get_time() * 0.05 * numbers::PI));
    } else {
      throw std::invalid_argument("Invalid component, neither 0 nor 1");
    }
  }
};

template <int dim>
SIDiffusionEquation<dim>::SIDiffusionEquation(int initial_param)
    : fe_system(FE_Q<dim>(1), 2), dof_handler(triangulation), time_step(1. / 4),
      time(time_step), timestep_number(1), theta(0.5), newton_stepsize(0.3),
      newton_threshold(0.02), diffusion_d(5e8), transmission_r(1.0),
      alpha_duration(0.22), initial_parameter(0) {
  initial_parameter = initial_param;
}

template <int dim> void SIDiffusionEquation<dim>::setup_system() {
  std::cout << "reading mesh" << std::endl;
  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f("Germany.msh");
  gridin.read_msh(f);
  triangulation.refine_global(1);

  // Below the faces on which the boundary-values are determined get selected.
  // only on the below boundaries the bondary conditions are valid. The
  // specified boundaries here will be ignored
  for (auto &face : triangulation.active_face_iterators()) {
    if (face->at_boundary()) {
      switch ((initial_parameter / 125) % 5) {
      case 0:
        if (face->center()[0] >= 3600000.) {
          face->set_boundary_id(1);
        }
        break;
      case 1:
        if (face->center()[0] <= 3600000.) {
          face->set_boundary_id(1);
        }
        break;
      case 2:
        if (face->center()[1] >= 5700000.) {
          face->set_boundary_id(1);
        }
        break;
      case 3:
        if (face->center()[1] <= 5600000.) {
          face->set_boundary_id(1);
        }
        break;
      case 4:
        if ((face->center()[0] <= 3600000.) &&
            (face->center()[1] <= 5700000.)) {
          face->set_boundary_id(1);
        }
        break;
      default:
        throw std::invalid_argument(
            "experiment identifier makes no sense while setting boundary ids");
      }
    }
  }

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe_system);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_euler_solution.reinit(dof_handler.n_dofs());
  old_euler_solution_add.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  euler_global_rhs.reinit(dof_handler.n_dofs());

  newton_update.reinit(dof_handler.n_dofs());
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
  constraints.condense(dsp);
}

// Solver for the linear system
template <int dim> void SIDiffusionEquation<dim>::solve() {
  SparseDirectUMFPACK A_direct;
  newton_update = system_rhs;
  A_direct.solve(system_matrix, newton_update);
}

// Saving the results as VTK and csv files
template <int dim> void SIDiffusionEquation<dim>::output_results() const {
  DataOut<dim> data_out;

  std::vector<std::string> solution_names(1, "susceptible");
  solution_names.emplace_back("infected");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          1, DataComponentInterpretation::component_is_scalar);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);
  std::ofstream output(filename);
  data_out.write_vtk(output);

  Vector<double> point_value(2);

  const std::string filename_h5 =
      "solution-" + std::to_string(initial_parameter) + "_" +
      Utilities::int_to_string(timestep_number, 3) + ".csv";
  std::ofstream csvfile;
  csvfile.open(filename_h5);
  csvfile << "Susceptible, Infected \n";

  std::vector<std::vector<double>> sample_points_coords;

  // Read the CSV file containing the coordinates of the evaluation points,
  // extract the X and Y coordinates
  // and store them in the "sample_points_coords" vector
  std::ifstream file("germany_centers.csv");

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
        if (i >= 1) // Extract X and Y coordinates from 3rd and 4th columns
        {
          coordinates.push_back(std::stod(token));
        }
      }
      sample_points_coords.push_back(coordinates);
    }
    file.close();
  } else {
    std::cerr << "Unable to open file points.csv" << std::endl;
    // Handle error if needed
  }
  // Evaluation of the solution at the given points
  // The value of the evaluation thes saved in a csvfile
  for (int i = 0; i < sample_points_coords.size(); i++) {
    try {
      VectorTools::point_value(
          dof_handler, solution,
          Point<2>(sample_points_coords[i][0], sample_points_coords[i][1]),
          point_value);
      csvfile << point_value[0] << ", " << point_value[1] << ""
              << "\n";
    } catch (...) {
      std::cout << "Not working on " << i << std::endl;
      std::cout << "Points there are:  " << sample_points_coords[i][0]
                << "and: " << sample_points_coords[i][1] << std::endl;
    }
  }
  csvfile.close();
}

template <int dim> void SIDiffusionEquation<dim>::run() {
  setup_system();

  // Set-up of the equation to solve for u.
  // for more information we refer to the documentation of deal.ii
  VectorTools::project(
      dof_handler, constraints, QGauss<dim>(fe_system.degree + 1),
      InitialValuesSusceptible<dim>(initial_parameter), old_euler_solution);

  VectorTools::project(
      dof_handler, constraints, QGauss<dim>(fe_system.degree + 1),
      InitialValuesSusceptible<dim>(initial_parameter), solution);

  for (; time <= 91; time += time_step, ++timestep_number) {
    std::cout << "Time step " << timestep_number << " at t=" << time
              << std::endl;
    // create the explizit part of euler stepping scheme:
    euler_global_rhs = 0;

    Vector<double> euler_cell_rhs(fe_system.n_dofs_per_cell());
    std::vector<types::global_dof_index> euler_local_dof_indices(
        fe_system.n_dofs_per_cell());

    FEValues<dim> euler_fe_values(fe_system, QGauss<dim>(fe_system.degree + 1),
                                  update_values | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);

    const FEValuesExtractors::Scalar susceptible_extr(0);
    const FEValuesExtractors::Scalar infected_extr(1);

    std::vector<Tensor<1, dim>> euler_old_susceptible_gradients(
        euler_fe_values.n_quadrature_points);
    std::vector<double> euler_old_susceptible_values(
        euler_fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> euler_old_infected_gradients(
        euler_fe_values.n_quadrature_points);
    std::vector<double> euler_old_infected_values(
        euler_fe_values.n_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      euler_cell_rhs = 0;
      euler_fe_values.reinit(cell);
      euler_fe_values[susceptible_extr].get_function_values(
          old_euler_solution, euler_old_susceptible_values);
      euler_fe_values[susceptible_extr].get_function_gradients(
          old_euler_solution, euler_old_susceptible_gradients);

      euler_fe_values[infected_extr].get_function_values(
          old_euler_solution, euler_old_infected_values);
      euler_fe_values[infected_extr].get_function_gradients(
          old_euler_solution, euler_old_infected_gradients);

      for (const unsigned int q : euler_fe_values.quadrature_point_indices()) {
        const double current_diff_coeff = Step23::diffusion_coefficient(
            euler_fe_values.quadrature_point(q), time, initial_parameter);
        const double current_transmission_r = r_value(
            euler_fe_values.quadrature_point(q), time, initial_parameter);

        for (const unsigned int i : euler_fe_values.dof_indices())
          euler_cell_rhs(i) +=
              ((euler_fe_values[susceptible_extr].value(i, q) *
                euler_old_susceptible_values[q]) -
               (euler_fe_values[susceptible_extr].value(i, q) * (1 - theta) *
                current_transmission_r * time_step *
                euler_old_susceptible_values[q] *
                euler_old_infected_values[q]) -
               (current_diff_coeff * (1 - theta) * time_step *
                euler_fe_values[susceptible_extr].gradient(i, q) *
                euler_old_susceptible_gradients[q])
               // second equation
               + (euler_fe_values[infected_extr].value(i, q) *
                  euler_old_infected_values[q]) +
               (time_step * (1 - theta) *
                current_transmission_r // transmission_r
                * euler_fe_values[infected_extr].value(i, q) *
                euler_old_susceptible_values[q] *
                euler_old_infected_values[q]) -
               (time_step * alpha_duration * (1 - theta) *
                euler_fe_values[infected_extr].value(i, q) *
                euler_old_infected_values[q]) -
               (time_step * current_diff_coeff * (1 - theta) *
                euler_fe_values[infected_extr].gradient(i, q) *
                euler_old_infected_gradients[q])) *
              euler_fe_values.JxW(q);
      }
      cell->get_dof_indices(euler_local_dof_indices);
      for (const unsigned int i : euler_fe_values.dof_indices())
        euler_global_rhs(euler_local_dof_indices[i]) += euler_cell_rhs(i);
    }

    // Set the boundary value correct
    BoundaryValuesSusceptible<dim> boundary_values_s_function(
        initial_parameter);
    boundary_values_s_function.set_time(time);

    if (time <= 20) {
      std::map<types::global_dof_index, double>
          boundary_values_between_timesteps;
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, boundary_values_s_function,
          boundary_values_between_timesteps);
      for (auto &boundary_value_loop : boundary_values_between_timesteps)
        solution(boundary_value_loop.first) = boundary_value_loop.second;
    }
    constraints.distribute(solution);

    double last_residual_norm = std::numeric_limits<double>::max();
    double newton_update_norm = std::numeric_limits<double>::max();
    do {
      system_matrix = 0;
      system_rhs = 0;
      const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();

      using ScratchData = MeshWorker::ScratchData<dim>;
      using CopyData = MeshWorker::CopyData<1, 1, 1>;
      using CellIteratorType = decltype(dof_handler.begin_active());

      const ScratchData sample_scratch_data(
          fe_system, QGauss<dim>(fe_system.degree + 1),
          update_values | update_gradients | update_quadrature_points |
              update_JxW_values);
      const CopyData sample_copy_data(dofs_per_cell);

      using ADHelper = Differentiation::AD::ResidualLinearization<
          Differentiation::AD::NumberTypes::sacado_dfad, double>;
      using ADNumberType = typename ADHelper::ad_type;

      const FEValuesExtractors::Scalar susceptible_fe(0);
      const FEValuesExtractors::Scalar infected_fe(1);

      const auto cell_worker = [&susceptible_fe, &infected_fe,
                                this](const CellIteratorType &cell,
                                      ScratchData &scratch_data,
                                      CopyData &copy_data) {
        const auto &fe_values = scratch_data.reinit(cell);
        const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();

        FullMatrix<double> &cell_matrix = copy_data.matrices[0];
        Vector<double> &cell_rhs = copy_data.vectors[0];
        std::vector<types::global_dof_index> &local_dof_indices =
            copy_data.local_dof_indices[0];
        cell->get_dof_indices(local_dof_indices);

        const unsigned int n_independent_variables = local_dof_indices.size();
        const unsigned int n_dependent_variables = dofs_per_cell;
        ADHelper ad_helper(n_independent_variables, n_dependent_variables);

        ad_helper.register_dof_values(solution, local_dof_indices);

        const std::vector<ADNumberType> &dof_values_ad =
            ad_helper.get_sensitive_dof_values();

        // Get susceptible values
        std::vector<ADNumberType> old_susceptible_values(
            fe_values.n_quadrature_points);
        fe_values[susceptible_fe].get_function_values_from_local_dof_values(
            dof_values_ad, old_susceptible_values);

        // Get susceptible gradients
        std::vector<Tensor<1, dim, ADNumberType>> old_susceptible_gradients(
            fe_values.n_quadrature_points);
        fe_values[susceptible_fe].get_function_gradients_from_local_dof_values(
            dof_values_ad, old_susceptible_gradients);

        // Get infected values
        std::vector<ADNumberType> old_infected_values(
            fe_values.n_quadrature_points);
        fe_values[infected_fe].get_function_values_from_local_dof_values(
            dof_values_ad, old_infected_values);

        // Get infected gradients
        std::vector<Tensor<1, dim, ADNumberType>> old_infected_gradients(
            fe_values.n_quadrature_points);
        fe_values[infected_fe].get_function_gradients_from_local_dof_values(
            dof_values_ad, old_infected_gradients);

        std::vector<ADNumberType> residual_ad(n_dependent_variables,
                                              ADNumberType(0.0));
        for (const unsigned int q : fe_values.quadrature_point_indices()) {
          const double current_diff_coeff_two = diffusion_coefficient(
              fe_values.quadrature_point(q), time, initial_parameter);
          const double current_transmission_r_two =
              r_value(fe_values.quadrature_point(q), time, initial_parameter);

          // const double current_diff_coeff_two = 1e8;
          // const double current_transmission_r_two = 1.0;
          for (const unsigned int i : fe_values.dof_indices()) {
            residual_ad[i] +=
                ((fe_values[susceptible_fe].value(i, q) *
                  old_susceptible_values[q]) +
                 (theta * time_step * current_transmission_r_two *
                  fe_values[susceptible_fe].value(i, q) *
                  old_susceptible_values[q] * old_infected_values[q]) +
                 (time_step * current_diff_coeff_two * theta *
                  fe_values[susceptible_fe].gradient(i, q) *
                  old_susceptible_gradients[q])
                 // second equation
                 +
                 (fe_values[infected_fe].value(i, q) * old_infected_values[q]) -
                 (theta * current_transmission_r_two * time_step *
                  fe_values[infected_fe].value(i, q) * old_infected_values[q] *
                  old_susceptible_values[q]) +
                 (time_step * alpha_duration * theta *
                  fe_values[infected_fe].value(i, q) * old_infected_values[q]) +
                 (time_step * current_diff_coeff_two * theta *
                  fe_values[infected_fe].gradient(i, q) *
                  old_infected_gradients[q])) *
                fe_values.JxW(q);
          }
        }
        ad_helper.register_residual_vector(residual_ad);
        ad_helper.compute_residual(cell_rhs);
        cell_rhs *= -1.0;
        ad_helper.compute_linearization(cell_matrix);
      };
      const auto copier = [dofs_per_cell, this](const CopyData &copy_data) {
        const FullMatrix<double> &cell_matrix = copy_data.matrices[0];
        const Vector<double> &cell_rhs = copy_data.vectors[0];
        const std::vector<types::global_dof_index> &local_dof_indices =
            copy_data.local_dof_indices[0];

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
      };

      MeshWorker::mesh_loop(dof_handler.active_cell_iterators(), cell_worker,
                            copier, sample_scratch_data, sample_copy_data,
                            MeshWorker::assemble_own_cells);

      system_rhs +=
          euler_global_rhs; // Jac * x = -F + const.  This is the constant

      constraints.condense(system_matrix);
      constraints.condense(system_rhs);

      // Do not update the value function on the border of the Domain!
      if (time <= 20) {
        std::map<types::global_dof_index, double> boundary_values;

        VectorTools::interpolate_boundary_values(
            dof_handler, 0, Functions::ZeroFunction<dim>(2), boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix,
                                           newton_update, system_rhs);
      }

      solve();
      /*calculate newton_residual*/
      last_residual_norm = system_rhs.l2_norm();
      newton_update_norm = newton_update.l2_norm();
      std::cout << "Last residual norm " << last_residual_norm
                << ". newton update l2-Norm: " << newton_update.l2_norm()
                << std::endl;
      solution.add(newton_stepsize, newton_update);
    } while (newton_update_norm > newton_threshold);

    output_results();
    old_euler_solution = solution;
  }
}
} // namespace Step23
int main(int argc, char *argv[]) {
  try {
    using namespace Step23;
    for (int i = 0; i < 125; i += 5) {
      SIDiffusionEquation<2> wave_equation_solver(i);
      wave_equation_solver.run();
    }
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
