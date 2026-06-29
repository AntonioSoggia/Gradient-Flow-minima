# Gradient Flow for Noisy Image Line Minimization

## Overview

This codebase implements **gradient descent optimization** to find **closed-form solutions of minima** for a noisy image that is fundamentally a line structure. The approach combines symbolic computation (via SymPy) with numerical optimization to analyze and find minima of a composite loss function in both 1D and 2D settings.

## Problem Statement

The core problem addressed by this code is:

> **Find closed-form solutions of minima in a noisy image that represents a line, under gradient descent.**

More precisely, the code seeks to minimize a composite objective function:

```
Objective = L + M
```

Where:
- **L**: A data fidelity term (sum of squared errors from a convolutional network output)
- **M**: A regularization term that models noise (σ²-weighted covariance terms)

The image is assumed to have a line-like structure, and the code explores how gradient descent flows converge to the minimizer set under different noise conditions (controlled by parameter σ²).

## Mathematical Formulation

### Variables

The optimization variables are:
- **E**: Encoding/weight matrix (3×3 in 2D, or 3-vector in 1D)
- **D**: Decoding/filter matrix (3×3 in 2D, or 3-vector in 1D)
- **σ²**: Noise variance parameter (controls the strength of regularization)

### Loss Components

1. **L (Data Fidelity Loss)**:
   - Computed via a mini-U-Net architecture (`Mini_Unet` class)
   - Involves convolution operations, pooling, and scattering
   - Measures deviation from a target (one-hot vector at position w₀)
   - `L = sum_of_squares(Y - target)` where Y is the network output

2. **M (Noise Regularization)**:
   - Models covariance structure of the noise
   - In 1D: M = (σ²/4) × [D² × (E² + E·E) + D₁D₃ × (E·E)]
   - In 2D: Complex quadratic form involving autocorrelation terms of E
   - Encourages smoothness and structure preservation

### Combined Objective

The code supports three gradient types:
- **"L"**: Minimize only the data fidelity term
- **"M"**: Minimize only the regularization term  
- **"combined"**: Minimize L + M (the full objective)
- **"sum"**: Alias for combined in some contexts

Additionally, there's a **combined mode** that averages objectives from even and odd w₀ values to improve numerical stability.

## Codebase Structure

```
Gradient_flow/
├── Function/                  # Core mathematical functions
│   ├── Sympy_functions.py    # Symbolic computation of L, M, gradients, Hessians
│   ├── Symbols.py            # Symbol definitions (E, D, σ², etc.)
│   ├── Matrix_Op_tot.py      # Mini-U-Net implementation with convolution operations
│   ├── Test_*.py             # Test files for numerical expressions
│   └── Test_Matrix.py        # Matrix operation tests
│
├── Gradient_Descent/         # Optimization module
│   ├── Gradient_Flow.py      # Main gradient descent implementation
│   ├── GF_res.py             # Result extraction and saving utilities
│   └── plot_cases.py         # Plotting utilities for visualization
│
├── Minimum/                  # Closed-form minimum analysis
│   ├── Min_computation.py    # Main minimum computation entry point
│   ├── Min_computation_helper.py  # Helper functions for solving gradient equations
│   └── Min_subs.py           # Substitution dictionaries for different configurations
│
├── Subs/                    # Substitution and simplification utilities
│   ├── GF_symplifie.py       # Gradient flow simplification
│   ├── GF_symplifie_helper.py
│   └── Subs.py
│
├── Regression/               # Regression analysis (separate concern)
│   ├── Regression.py
│   ├── Stats_methods.py
│   └── File_processing.py
│
└── *.txt                    # Result files from various experiments
```

## Key Classes and Functions

### `FunctionSimpy` (Sympy_functions.py)

The central class that computes symbolic expressions for the objective and its derivatives.

```python
class FunctionSimpy:
    # Computes L, M, L+M symbolically
    # Can compute gradients and Hessians
    # Supports numeric evaluation via lambdification
    
    def __init__(self, use_Numeric, use_1D, subs, w0, sigma2_value, 
                 combined=False, w0_even=None, w0_odd=None)
    
    def compute_L()     # Returns symbolic L expression
    def compute_M()     # Returns symbolic M expression
    def evaluate(vals, func_type)  # Numeric evaluation
    def compute_gradient(expr, vars)  # Symbolic gradient
    def compute_hessian(expr, vars)   # Symbolic Hessian
```

### `Mini_Unet` (Matrix_Op_tot.py)

A minimal U-Net-like architecture for computing the forward pass:

```python
class Mini_Unet:
    def conv_valid_sym(matrix, filt)      # Symbolic convolution
    def pool(X)                           # Pooling operation
    def scatter_pool(A)                  # Scatter/upsample
    def pad_matrix(matrix, pad)          # Padding
    def ones()                           # Create target matrix
    def compute_Y(subs, same_size)       # Full forward pass
    def sum_of_squares(Y, target)        # Compute L loss
```

### `GradientDescent` (Gradient_Flow.py)

Implements gradient descent optimization:

```python
class GradientDescent:
    def __init__(self, config)           # Initialize with parameters
    def loss_gradients(E, D)             # Compute gradient at point
    def simulate_gradient_flow(params)     # Single gradient descent run
    def safe_simulate_gradient_flow()     # Robust version with error handling
    def parallel_gradient_flow()          # Parallel optimization from multiple starts
    def plot_gradient_flow()             # Visualize convergence
    def compute_value()                  # Evaluate objective at points
```

### Minimum Computation (Minimum/ directory)

Provides tools for finding **closed-form solutions** by solving ∇(L+M) = 0:

```python
# Min_computation_helper.py
def solve_for_D(grad_vec, D_vars)        # Solve linear system for D variables
def gershgorin_test(matrix)             # Check positive definiteness via Gershgorin
def ldl_test(H)                         # LDL decomposition for definite testing

def computation_helper(...)             # Main helper for computing gradients/Hessians
```

## The Minimizer Set

The theoretical minimizer set for the combined objective (L + M) forms specific curves:

### In 1D (use_1D=True):
- For combined mode: `D₂ = 1 / (E₁ + E₂)` or `D₂ = 1 / ((σ² + 1) × (E₁ + E₂))`
- The minimizer set is a hyperbola in the (E, D) space

### In 2D (use_1D=False):
- The minimizer set satisfies: `D₂₂ = 1 / ΣEᵢⱼ` (sum over specific E entries)
- Forms a curve in the high-dimensional (E, D) space

These closed-form solutions are derived in the `Minimum/` directory by solving the gradient equations symbolically.

## Gradient Descent Dynamics

The code studies how gradient descent trajectories converge to the minimizer set:

```python
# Typical configuration
config = {
    "eta": 1e-6,           # Learning rate
    "steps": 10000,        # Maximum iterations
    "tol": 1e-5,          # Convergence tolerance
    "n_points": 10,       # Number of random initializations
    "E_range": (0.5, 0.6), # Initial E range
    "D_range": (0.5, 0.6), # Initial D range
    "sigma2": 0,          # Noise level (0 = no noise)
    "grad_type": "combined",
    "full_trajectory": True,  # Save full trajectory
    "use_1D": False,      # Use 2D mode
    "w0": 6,             # Target position in one-hot vector
    "subs": 0            # Substitution level for simplification
}

grad = GradientDescent(config)
grad.plot_gradient_flow()  # Visualize convergence to minimizer set
```

## Visualization

The `plot_gradient_flow()` method generates plots showing:
- The theoretical minimizer set (green curves)
- Initial random points (blue crosses)
- Final converged points (black dots)
- Full trajectories (black lines, if `full_trajectory=True`)
- Error points that stayed at initialization (red squares)

Example plot labels:
- x-axis: Sum of E entries (e.g., `E₁₁ + E₁₂ + ...`)
- y-axis: Specific D entries (e.g., `D₂₂`)

## Noise Analysis

The `sigma2` parameter controls the noise level:
- **σ² = 0**: No noise, pure line structure
- **σ² > 0**: Noisy image, stronger regularization

The code in `GF_res.py` runs experiments across different σ² values and saves:
- Final objective values
- Full checkpoint trajectories
- Convergence statistics

## Closed-Form Solutions

The `Minimum/` directory attempts to find **analytical closed-form solutions** by:

1. Computing the gradient ∇(L+M) symbolically
2. Setting each component to zero
3. Solving the resulting system of equations
4. Applying appropriate substitutions to simplify

For the 2D case with combined mode, the solution for D variables can be expressed as:
```
D₂₂ = 1 / (E₁₁ + E₂₁ + E₃₁ + E₁₂ + E₂₂ + E₃₂)
```

For the 1D case:
```
E₁ = 1 / (D₂ × (σ² + 1))
E₂ = 0
E₃ = 0
D₁ = 0
D₃ = D₂
```

These solutions represent the **minimizer set** that gradient descent trajectories converge to.

## Usage Examples

### Run Gradient Descent Visualization
```bash
cd Gradient_Descent
python Gradient_Flow.py
```

### Extract Results with Checkpoints
```bash
python GF_res.py
```

### Compute Closed-Form Minima
```bash
cd Minimum
python Min_computation.py
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `eta` | Learning rate | 1e-6 to 5e-3 |
| `steps` | Max iterations | 10000-25000 |
| `tol` | Gradient norm tolerance | 1e-5 to 1e-6 |
| `n_points` | Number of initializations | 1-10 |
| `sigma2` | Noise variance | 0, 1, 2, ... |
| `w0` | Target position | 5, 6, ... |
| `use_1D` | Dimensionality | True/False |
| `grad_type` | Objective type | "L", "M", "combined" |
| `full_trajectory` | Save all points | True/False |
| `subs` | Simplification level | 0, 1, 2 |
| `combined` | Average even/odd w0 | True/False |

## Results Interpretation

The code produces several types of results:

1. **Convergence Plots**: Show trajectories converging to minimizer curves
2. **Final Objectives**: Sorted list of (E, D, L, M, L+M) at convergence
3. **Checkpoint Files**: Full trajectories saved as CSV for analysis
4. **Closed-Form Solutions**: Symbolic expressions for minima

Example output:
```
E = [0.1667, 0.1667, 0.1667], D = [0.0, 1.5, 0.0], S = 0, 
L = 0.0000, M = 0.0000, L+M = 0.0000
```

## Theoretical Insights

The minimizer set forms a **manifold** in the (E, D) space. The dimension of this manifold:
- In 1D: 1-dimensional curve (hyperbola)
- In 2D: Higher-dimensional manifold

The gradient flow converges to this manifold from almost all initial conditions, with convergence rate depending on:
- The conditioning of the Hessian
- The learning rate η
- The noise level σ²

The Hessian analysis (via `ldl_test()` and `gershgorin_test()`) confirms that the objective is convex in certain subspaces, ensuring convergence to the minimizer set.

---

## Mathematical Background

The problem relates to **structured optimization** in inverse problems. The line image assumption means the underlying signal has low-dimensional structure, which the regularization term M captures. The gradient flow analysis shows how different initializations converge to the same minimizer manifold, demonstrating that the optimization landscape has favorable geometry despite the non-convexity of individual terms.

### Connection to Literature

This work relates to:
- **Gradient flow in non-convex optimization**
- **Structured sparsity and low-rank recovery**
- **Convolutional neural networks for inverse problems**
- **Manifold optimization**

The closed-form solutions provide insight into why gradient descent works well for this class of problems despite non-convexity.

---

## Citation

If using this code for research, please reference the repository and any associated publications.

```bibtex
@misc{gradient_flow,
  author = {Antonio},
  title = {Gradient Flow for Noisy Line Image Minimization},
  year = {2025},
  howpublished = {\url{https://github.com/antonio/Gradient_flow}},
}
```

---

## License

This code is provided as-is for research purposes. Contact the author for commercial use.
