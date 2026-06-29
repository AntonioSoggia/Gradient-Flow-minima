import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from Function.Sympy_functions import FunctionSimpy
import sympy as sp
from plot_cases import *
import os
from dotenv import load_dotenv


class GradientDescent:
    def __init__(self, config):
        self.eta = config["eta"]
        self.steps = config["steps"]
        self.tol = config["tol"]
        self.n_points = config["n_points"]
        self.E_range = config["E_range"]
        self.D_range = config["D_range"]
        self.sigma2 = float(config["sigma2"])
        self.grad_type = config["grad_type"]
        self.w0 = config["w0"]
        self.full_trajectory = config["full_trajectory"]
        self.use_1D = config.get("use_1D", True)
        self.subs = config["subs"]
        if self.grad_type == "combined":
            self.combined = True
            self.w0_even = config.get("w0_even", 5)
            self.w0_odd = config.get("w0_odd", 6)
            self.func_calc = FunctionSimpy(use_Numeric=True, use_1D=self.use_1D, subs=self.subs, w0=self.w0,
                                           sigma2_value=self.sigma2, combined=True,
                                           w0_even=self.w0_even, w0_odd=self.w0_odd)
            avg_obj = sp.simplify(self.func_calc.compute_L() + self.func_calc.compute_M())
            print("Combined objective (average of even and odd):", avg_obj)
        else:
            self.combined = False
            self.w0 = config.get("w0", 6)
            self.func_calc = FunctionSimpy(use_Numeric=True, use_1D=self.use_1D, subs=self.subs, w0=self.w0,
                                           sigma2_value=self.sigma2)
            obj = sp.simplify(self.func_calc.compute_L() + self.func_calc.compute_M())
            print("Objective:", obj)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "func_calc" in state:
            del state["func_calc"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.combined:
            self.func_calc = FunctionSimpy(use_Numeric=True, use_1D=self.use_1D, subs=self.subs, w0=self.w0,
                                           sigma2_value=self.sigma2, combined=True,
                                           w0_even=self.w0_even, w0_odd=self.w0_odd)
        else:
            self.func_calc = FunctionSimpy(use_Numeric=True, use_1D=self.use_1D, subs=self.subs, w0=self.w0,
                                           sigma2_value=self.sigma2)

    def generate_random_initializations(self):
        if self.use_1D:
            random_E = [np.random.uniform(*self.E_range, 3) for _ in range(self.n_points)]
            random_D = [np.random.uniform(*self.D_range, 3) for _ in range(self.n_points)]
        else:
            random_E = [np.random.uniform(*self.E_range, (3, 3)) for _ in range(self.n_points)]
            random_D = [np.random.uniform(*self.D_range, (3, 3)) for _ in range(self.n_points)]
        return list(zip(random_E, random_D))

    def loss_gradients(self, E, D):
        # Build a tuple for the optimization variables (E and D only).
        if self.use_1D:
            vals_opt = tuple(E) + tuple(D)
            nE = 3
        else:
            vals_opt = tuple(E.flatten()) + tuple(D.flatten())
            nE = 9

        # In combined mode, use the evaluate key 'grad_combined' which returns the averaged gradient.
        if self.combined:
            grad = np.array(self.func_calc.evaluate(vals_opt, func_type='grad_combined')).flatten()
        else:
            grad = np.array(self.func_calc.evaluate(vals_opt, func_type='grad_' + self.grad_type)).flatten()

        grad_E = grad[:nE]
        grad_D = grad[nE:]
        if self.use_1D:
            return grad_E, grad_D
        else:
            return grad_E.reshape((3, 3)), grad_D.reshape((3, 3))

    def simulate_gradient_flow(self, params, checkpoint_interval=None):
        init_E, init_D = params
        E = np.array(init_E, dtype=float)
        D = np.array(init_D, dtype=float)

        # Initialize trajectory list for checkpoints
        trajectory = []
        if checkpoint_interval is not None and checkpoint_interval > 0:
            trajectory.append((E.copy(), D.copy()))  # Save initial point as first checkpoint

        for step in range(self.steps):
            grad_E, grad_D = self.loss_gradients(E, D)
            # Stop if the norm of the full gradient is less than the tolerance.
            if np.linalg.norm(np.concatenate([grad_E.flatten(), grad_D.flatten()])) < self.tol:
                break
            E_new = E - self.eta * grad_E
            D_new = D - self.eta * grad_D
            if (np.any(np.isnan(E_new)) or np.any(np.isinf(E_new)) or
                    np.any(np.isnan(D_new)) or np.any(np.isinf(D_new))):
                raise RuntimeError("Bad values encountered during gradient descent.")
            E, D = E_new, D_new

            # Save checkpoint if interval is met
            if checkpoint_interval is not None and checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                trajectory.append((E.copy(), D.copy()))

        # --- RETURN TYPE BASED ON CHECKPOINTING AND FULL_TRAJECTORY ---
        if checkpoint_interval is not None and checkpoint_interval > 0:
            # If checkpointing is active, always return the full trajectory list
            if not trajectory or not (np.array_equal(trajectory[-1][0], E) and np.array_equal(trajectory[-1][1], D)):
                trajectory.append((E.copy(), D.copy()))
            return trajectory
        else:
            # If no checkpoint interval
            if self.full_trajectory:
                # If full_trajectory is True, ensure the initial point is included and return the full trajectory list
                if not trajectory:  # This means no checkpoints were added, but full_trajectory is True
                    trajectory.append((init_E.copy(), init_D.copy()))  # Add initial point
                # Ensure the final point is included if it's not already the last one
                if not trajectory or not (
                        np.array_equal(trajectory[-1][0], E) and np.array_equal(trajectory[-1][1], D)):
                    trajectory.append((E.copy(), D.copy()))
                return trajectory
            else:
                # If full_trajectory is False and no checkpoint_interval, return just the final (E, D) tuple
                return (E, D)  # Return a tuple, not a list

    def safe_simulate_gradient_flow(self, params, checkpoint_interval=None):
        init_E, init_D = params
        initial_point_tuple = (np.array(init_E, dtype=float), np.array(init_D, dtype=float))
        try:
            # Pass checkpoint_interval to simulate_gradient_flow
            result = self.simulate_gradient_flow(params, checkpoint_interval=checkpoint_interval)
        except Exception as e:
            print("Simulation error:", e)
            if self.full_trajectory:
                return ([(np.array(init_E, dtype=float), np.array(init_D, dtype=float))],
                        True,
                        (np.array(init_E, dtype=float), np.array(init_D, dtype=float)))
            else:
                return ((np.array(init_E, dtype=float), np.array(init_D, dtype=float)),
                        True,
                        (np.array(init_E, dtype=float), np.array(init_D, dtype=float)))
        else:
            return result, False, (np.array(init_E, dtype=float), np.array(init_D, dtype=float))

    @staticmethod  # ADD THIS DECORATOR
    def _wrapper_safe_simulate_static(instance_and_params):  # Rename to a static method
        """
        Wrapper function for multiprocessing.Pool to call safe_simulate_gradient_flow.
        This needs to be a static method or a top-level function to be picklable.
        """
        instance, params, checkpoint_interval = instance_and_params
        return instance.safe_simulate_gradient_flow(params, checkpoint_interval)

    def parallel_gradient_flow(self, checkpoint_interval=None):  # ADD checkpoint_interval parameter
        init_points = self.generate_random_initializations()
        num_workers = min(cpu_count(), self.n_points)

        # Prepare arguments for multiprocessing map
        # Each item in `map` needs to be a single argument, so we pass a tuple
        # (self, params, checkpoint_interval) to the static wrapper.
        map_args = [(self, params, checkpoint_interval) for params in init_points]

        with Pool(num_workers) as p:
            # Call the static wrapper method
            results_with_trajectories = p.map(self._wrapper_safe_simulate_static, map_args)
        return results_with_trajectories

    def plot_gradient_flow(self):
        len_points = 200
        # `results` is a list of tuples: (simulation_output, error_flag, initial_parameters)
        # Note: plot_gradient_flow will now receive trajectories, so it might need adjustments
        # if it expects only final E,D points for plotting.
        # For now, we'll call parallel_gradient_flow without checkpointing for plotting
        # to get only the final points if full_trajectory is False.
        # If full_trajectory is True, it will still get the full trajectory.
        results = self.parallel_gradient_flow(
            checkpoint_interval=None)  # Pass None for plotting to get final points or full trajectory based on self.full_trajectory
        plt.figure(figsize=(10, 8))
        curves, xlabel, ylabel = get_curve_data(self.grad_type, self.use_1D, self.sigma2, len_points)
        for (X, Y, style, label) in curves:
            plt.plot(X, Y, style, label=label)
        x_success, y_success, x_init, y_init, x_error, y_error = get_plot_data(
            results, self.use_1D, self.grad_type, self.full_trajectory
        )
        if self.full_trajectory:
            for i in range(len(x_success)):
                if x_success[i] and y_success[i]:
                    plt.plot(x_success[i], y_success[i], color='black', lw=0.8)
        else:
            if x_success and y_success:
                plt.scatter(x_success, y_success, s=10, marker='o', label='Final Converged Points')
        if x_init and y_init:
            plt.scatter(x_init, y_init, s=10, marker='x', color='blue',
                        label='Initial Points')
        if x_error and y_error:
            plt.scatter(x_error, y_error, color='red', s=10, marker='s',
                        label='Error: Ended at Init.')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title('Gradient Flow Converging to the Minimizer Set')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0, 7)
        plt.ylim(0, 2)
        plt.legend()
        plt.grid()
        plt.savefig('GradientFlow_det.pdf')  # Uncomment to save
        plt.show(block=False)

    def compute_value(self, simulation_results_list, x_min=1):
        results_with_loss = []
        for sim_output_trajectory, error_flag, init_params in simulation_results_list:  # sim_output is now always a trajectory
            E_eval, D_eval = None, None
            if error_flag:
                E_eval, D_eval = init_params  # Use initial parameters if error occurred
            else:  # No error, use the last point of the trajectory
                if sim_output_trajectory and isinstance(sim_output_trajectory, list):
                    E_eval, D_eval = sim_output_trajectory[-1]  # Get the last point of the trajectory
                else:
                    # Fallback if trajectory is empty/invalid for a successful run
                    print(
                        f"Warning: Successful run with invalid trajectory. Using initial params: {init_params}")
                    E_eval, D_eval = init_params

            if E_eval is None or D_eval is None:
                print(
                    f"Could not determine E, D for evaluation from: {(sim_output_trajectory, error_flag, init_params)}. Skipping.")
                continue

            if self.use_1D:
                vals = tuple(E_eval.tolist() + D_eval.tolist())
            else:
                vals = tuple(E_eval.flatten().tolist() + D_eval.flatten().tolist())
            L_val = self.func_calc.evaluate(vals, func_type='L')
            M_val = self.func_calc.evaluate(vals, func_type='M')
            results_with_loss.append({'E': E_eval, 'D': D_eval, 'S': self.sigma2,
                                      'L': L_val, 'M': M_val, 'L+M': L_val + M_val})

        sorted_results = sorted(results_with_loss, key=lambda x: x['L+M'])
        minimal_results_data = sorted_results[:x_min]
        formatted_results = [
            f"E = {res['E']}, D = {res['D']}, S = {res['S']}, L = {res['L']:.4f}, M = {res['M']:.4f}, L+M = {res['L+M']:.4f}"
            for res in minimal_results_data
        ]
        return formatted_results


def load_config_from_env():
    """Load configuration from environment variables with defaults."""
    load_dotenv()  # Load from .env file
    
    return {
        "eta": float(os.getenv("ETA", "1e-6")),
        "steps": int(os.getenv("STEPS", "10000")),
        "tol": float(os.getenv("TOL", "1e-5")),
        "n_points": int(os.getenv("N_POINTS", "10")),
        "E_range": (float(os.getenv("E_RANGE_MIN", "0.5")), float(os.getenv("E_RANGE_MAX", "0.6"))),
        "D_range": (float(os.getenv("D_RANGE_MIN", "0.5")), float(os.getenv("D_RANGE_MAX", "0.6"))),
        "sigma2": float(os.getenv("SIGMA2", "0")),
        "grad_type": os.getenv("GRAD_TYPE", "combined"),
        "full_trajectory": os.getenv("FULL_TRAJECTORY", "True").lower() == "true",
        "use_1D": os.getenv("USE_1D", "False").lower() == "true",
        "w0_even": int(os.getenv("W0_EVEN", "5")),
        "w0_odd": int(os.getenv("W0_ODD", "6")),
        "w0": int(os.getenv("W0", "6")),
        "subs": int(os.getenv("SUBS", "0"))
    }


if __name__ == '__main__':
    # Load configuration from environment variables or use defaults
    config = load_config_from_env()
    
    print("Loaded configuration:")
    print(f"  eta: {config['eta']}")
    print(f"  steps: {config['steps']}")
    print(f"  sigma2: {config['sigma2']}")
    print(f"  grad_type: {config['grad_type']}")
    print(f"  use_1D: {config['use_1D']}")
    grad = GradientDescent(config)
    print("Plotting gradient flow...")
    grad.plot_gradient_flow()
    print("\nComputing final objectives...")
    final_points_info = grad.parallel_gradient_flow()
    num_results_to_print = min(7, len(final_points_info))
    if final_points_info:
        final_objective_strings = grad.compute_value(final_points_info, x_min=num_results_to_print)
        print(f"\nTop {len(final_objective_strings)} final objectives (sorted by L+M):")
        for res_str in final_objective_strings:
            print(res_str)
    else:
        print("No simulation results to compute objectives from.")
    if plt.get_fignums():
        print("\nDisplaying plot. Close plot window to exit.")
        plt.show(block=True)
    else:
        print("\nNo plot to display or plot closed.")