import numpy as np
import pandas as pd
import os
from Gradient_Flow import GradientDescent

# Assuming GradientDescent, FunctionSimpy, etc. are imported or defined above this function

def extract_and_save_results(checkpoint_interval=5000):
    """
    Extracts gradient descent results and saves final objectives and checkpoints.

    Args:
        checkpoint_interval (int): The number of steps between each checkpoint save.
                                   Set to 0 or None to only save the final point.
    """
    all_final_objectives = []
    all_checkpoints_data = []  # Stores flattened checkpoint data for all simulations

    # Determine point_dimension based on use_1D from config
    # This assumes a consistent dimension across all simulations, which is usually the case.
    # We'll get it from the first config iteration.
    point_dimension = 0

    for i in range(1, 3):  # Loop through sigma2 values from 1 to 15
        config = {
            "eta": 5e-3,
            "steps": 25000,
            "tol": 1e-6,
            "n_points": 1,  # Reduced number of parallel points for quicker demonstration
            "E_range": (1, 1.1),
            "D_range": (1, 1.1),
            "sigma2": i,
            "grad_type": "combined",
            "full_trajectory": True,
            # This flag in config now only affects internal logic if you still use it for other purposes,
            # but the checkpointing will always return a trajectory.
            "use_1D": False,
            "w0_even": 5,
            "w0_odd": 6,
            "w0": 6,
            "subs": 0
        }

        grad = GradientDescent(config)

        # Determine point_dimension dynamically from the first run's config
        if point_dimension == 0:
            if config["use_1D"]:
                point_dimension = 3 + 3  # 3 E values + 3 D values
            else:
                point_dimension = 9 + 9  # 9 E values + 9 D values (3x3 matrices)

        # Call parallel_gradient_flow with the specified checkpoint_interval
        # final_points_info now contains (trajectory, error_flag, initial_value)
        sim_results_with_trajectories = grad.parallel_gradient_flow(checkpoint_interval=checkpoint_interval)

        for trajectory, error_flag, init_val in sim_results_with_trajectories:
            # Flatten each point (E, D tuple) in the trajectory and combine into a single row
            combined_row = []
            for E_point, D_point in trajectory:
                combined_row.extend(E_point.flatten())
                combined_row.extend(D_point.flatten())
            all_checkpoints_data.append(combined_row)

            # --- Original logic for final objectives (still needed) ---
            # The last point in the trajectory is considered the final point for evaluation
            # compute_value expects a list of (sim_output, error_flag, init_params)
            # where sim_output is the trajectory, and it will extract the last point.
            # We need to pass a list containing just this one simulation's result.
            loss_vals = grad.compute_value([(trajectory, error_flag, init_val)])
            all_final_objectives.extend(loss_vals)
            # ---------------------------------------------------------

    # --- Save final objectives to the original file ---
    output_file_objectives = "/home/antonio/Desktop/Notebooks/PythonProject/Regression/gradient_descent_results_sigma_mix_TDA.csv"
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_objectives), exist_ok=True)
    with open(output_file_objectives, "w") as f:
        for res in all_final_objectives:
            f.write(res + "\n")
    print(f"Final objectives saved to: {output_file_objectives}")

    output_file_checkpoints = "TDA.csv"
    max_cols = 0
    if all_checkpoints_data:
        max_cols = max(len(row) for row in all_checkpoints_data)
    column_names = []
    if point_dimension > 0:
        max_checkpoints_in_data = max_cols // point_dimension
        for cp_idx in range(max_checkpoints_in_data):
            for dim_idx in range(point_dimension):
                column_names.append(f"checkpoint_{checkpoint_interval * cp_idx}_dim_{dim_idx + 1}")

    df_checkpoints = pd.DataFrame(all_checkpoints_data)

    if not df_checkpoints.empty and column_names:
        df_checkpoints.columns = column_names[:df_checkpoints.shape[1]]

    # Save the DataFrame to a CSV file
    df_checkpoints.to_csv(output_file_checkpoints, index=False)
    print(f"Checkpoints saved to: {output_file_checkpoints}")

if __name__== "__main__":
    extract_and_save_results()