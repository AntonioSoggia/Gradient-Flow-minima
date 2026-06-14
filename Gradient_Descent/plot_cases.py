import numpy as np

def process_1D_results(results, full_trajectory_flag): # Added full_trajectory_flag
    """
    Process simulation results for 1D.
    Accepts full_trajectory_flag to handle trajectories or final points.
    Returns:
        x_success, y_success, x_init, y_init, x_error, y_error
    """
    x_success, y_success = [], []
    x_init, y_init = [], []
    x_error, y_error = [], []

    for sim_output, error_flag, init_val_tuple in results:
        init_E, init_D = init_val_tuple

        # Populate initial points for every simulation run
        x_init.append(init_E[0] + init_E[1])
        y_init.append(init_D[1])

        if error_flag:
            x_error.append(init_E[0] + init_E[1]) # Store initial point as error point
            y_error.append(init_D[1])
            if full_trajectory_flag: # Maintain structure for success lists
                x_success.append([]) # Append empty list for this failed trajectory
                y_success.append([])
        else:
            if full_trajectory_flag:
                current_traj_x = []
                current_traj_y = []
                if isinstance(sim_output, list) and sim_output:
                    for E_step, D_step in sim_output:
                        current_traj_x.append(E_step[0] + E_step[1])
                        current_traj_y.append(D_step[1])
                else:
                    print(f"Warning (1D): Expected a trajectory list for successful run, got {type(sim_output)}. Plotting empty.")
                x_success.append(current_traj_x)
                y_success.append(current_traj_y)
            else:
                try:
                    final_E, final_D = sim_output
                    x_success.append(final_E[0] + final_E[1])
                    y_success.append(final_D[1])
                except (TypeError, ValueError) as e:
                    print(f"Warning (1D): Could not unpack final E, D from sim_output: {sim_output}. Error: {e}. Skipping success plot for this point.")

    return x_success, y_success, x_init, y_init, x_error, y_error
def process_2D_results(results, grad_type, full_trajectory_flag): # Added full_trajectory_flag
    """
    Process simulation results for 2D.
    Accepts full_trajectory_flag to handle trajectories or final points.
    Returns:
        x_success, y_success, x_init, y_init, x_error, y_error
    """
    x_success, y_success = [], []
    x_init, y_init = [], []
    x_error, y_error = [], []

    for sim_output, error_flag, init_val_tuple in results:
        init_E, init_D = init_val_tuple

        # Determine initial x, y based on grad_type
        if grad_type in ['sum', 'combined']:
            current_x_init = (init_E[0,0] + init_E[0,1] +
                              init_E[1,0] + init_E[1,1] +
                              init_E[2,0] + init_E[2,1])
            current_y_init = init_D[1,1] + init_D[0, 1] +  init_D[2, 1]
        else:
            current_x_init = init_E[0,0] + init_E[0,1]
            current_y_init = init_D[0,1]

        x_init.append(current_x_init)
        y_init.append(current_y_init)

        if error_flag:
            x_error.append(current_x_init)
            y_error.append(current_y_init)
            if full_trajectory_flag:
                x_success.append([])
                y_success.append([])
        else:
            if full_trajectory_flag:
                current_traj_x = []
                current_traj_y = []
                if isinstance(sim_output, list) and sim_output:
                    for E_step, D_step in sim_output:
                        if grad_type in ['sum', 'combined']:
                            x_coord = (E_step[0,0] + E_step[0,1] +
                                       E_step[1,0] + E_step[1,1] +
                                       E_step[2,0] + E_step[2,1])
                            y_coord = D_step[1,1]
                        else:
                            x_coord = E_step[0,0] + E_step[0,1]
                            y_coord = D_step[0,1]
                        current_traj_x.append(x_coord)
                        current_traj_y.append(y_coord)
                else:
                     print(f"Warning (2D): Expected a trajectory list for successful run, got {type(sim_output)}. Plotting empty.")
                x_success.append(current_traj_x)
                y_success.append(current_traj_y)
            else:
                try:
                    final_E, final_D = sim_output
                    if grad_type in ['sum', 'combined']:
                        x_coord = (final_E[0,0] + final_E[0,1] +
                                   final_E[1,0] + final_E[1,1] +
                                   final_E[2,0] + final_E[2,1])
                        y_coord = final_D[1,1]
                    else:
                        x_coord = final_E[0,0] + final_E[0,1]
                        y_coord = final_D[0,1]
                    x_success.append(x_coord)
                    y_success.append(y_coord)
                except (TypeError, ValueError) as e:
                    print(f"Warning (2D): Could not unpack final E, D from sim_output: {sim_output}. Error: {e}. Skipping success plot for this point.")

    return x_success, y_success, x_init, y_init, x_error, y_error

def get_plot_data(results, use_1D, grad_type, full_trajectory_flag): # Added full_trajectory_flag
    if use_1D:
        return process_1D_results(results, full_trajectory_flag)
    else:
        return process_2D_results(results, grad_type, full_trajectory_flag)

def get_curve_data(grad_type, use_1D, sigma2, len_points):
    curves = []
    xlabel, ylabel = 'x', 'y'
    if grad_type == 'L':
        if use_1D:
            X_positive = np.linspace(0.1, 10, int(len_points/2))
            X_negative = np.linspace(-10, -0.1, int(len_points/2))
            Y_positive = 2.0 / X_positive
            Y_negative = 2.0 / X_negative
            curves.append((X_positive, Y_positive, 'r-', 'L minimizer (Positive)'))
            curves.append((X_negative, Y_negative, 'r-', 'L minimizer (Negative)'))
            xlabel, ylabel = '$E_1+E_2$', '$D_2$'
        else:
            # Define 2D curves if needed
            xlabel, ylabel = 'x', 'y'
    elif grad_type in ['sum', 'combined']:
        if use_1D:
            X_positive = np.linspace(0.1, 10, int(len_points/2))
            X_negative = np.linspace(-10, -0.1, int(len_points/2))
            Y_positive = 1.0 / (X_positive * (sigma2 + 1))
            Y_negative = 1.0 / (X_negative * (sigma2 + 1))
            curves.append((X_positive, Y_positive, 'g-', 'Minimizer set'))
            curves.append((X_negative, Y_negative, 'g-', 'Minimizer set'))
            xlabel, ylabel = '$D_2$', '$E_1$'
        else:
            X_curve = np.linspace(0.1, 10, len_points)
            Y_curve = 1 / X_curve
            curves.append((X_curve, Y_curve, 'g-', 'Minimizer set'))
            xlabel, ylabel = r'$\sum_{i=1}^3  E_{i,1}$', r'$D_{22}$'
    elif grad_type == 'M':
        X_positive = np.linspace(0.1, 10, len_points)
        Y_placeholder = np.zeros_like(X_positive)
        curves.append((X_positive, Y_placeholder, 'p-', 'M minimizer placeholder'))
        xlabel, ylabel = '$x$', '$y$'
    return curves, xlabel, ylabel