import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import time

def plot_function_surface(func, bounds, results_dict, dims_to_plot, title="Function Surface",
                          clip_z=True, log_scale=False, extra_dims_vals=None, 
                          grid_size=300, history_dict=None, show_history=False):
    """
    3D surface plot for higher-dimensional functions with projections onto the base plane
    and optional optimization history trails.

    Parameters:
    - func: function to evaluate
    - bounds: list of tuples [(min, max), ...] for each dimension
    - results_dict: dict of {'Algorithm': solution_vector}
    - dims_to_plot: tuple/list of 2 indices specifying which dimensions to plot
    - extra_dims_vals: values for other dimensions not plotted
    - clip_z: clip extreme Z values
    - log_scale: apply log1p transformation to Z
    - grid_size: resolution of the mesh
    - history_dict: optional dict {'Algorithm': list_of_points} to show search history
    - show_history: bool, if True plots optimization history trails
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure extra_dims_vals exists
    if extra_dims_vals is None:
        extra_dims_vals = [b[1] for b in bounds]

    # Create meshgrid
    x = np.linspace(bounds[dims_to_plot[0]][0], bounds[dims_to_plot[0]][1], grid_size)
    y = np.linspace(bounds[dims_to_plot[1]][0], bounds[dims_to_plot[1]][1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute function values
    for i in range(grid_size):
        for j in range(grid_size):
            full_x = []
            for d in range(len(bounds)):
                if d == dims_to_plot[0]:
                    full_x.append(X[i,j])
                elif d == dims_to_plot[1]:
                    full_x.append(Y[i,j])
                else:
                    full_x.append(extra_dims_vals[d])
            Z[i,j] = func(full_x)

    # Clip or log-scale
    if clip_z:
        Z_max = np.percentile(Z, 95)
        Z = np.clip(Z, np.min(Z), Z_max)
    if log_scale:
        Z = np.log1p(Z - np.min(Z))

    z_base = np.min(Z)

    # Plot surface
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, zdir='z', offset=z_base, cmap='viridis', alpha=0.7)

    colors = ['red','blue','green','orange','purple']

    for i, (label, sol) in enumerate(results_dict.items()):
        # Prepare full solution vector
        full_sol = [sol[d] if d in dims_to_plot else extra_dims_vals[d] for d in range(len(bounds))]
        z_val = func(full_sol)
        if clip_z:
            z_val = min(z_val, Z_max)
        if log_scale:
            z_val = np.log1p(z_val - np.min(Z))

        # 3D solution marker
        ax.scatter(sol[dims_to_plot[0]], sol[dims_to_plot[1]], z_val,
                   color=colors[i % len(colors)], s=50,
                   label=f"{label}: {func(sol):.4f}")
        # Projection marker on base
        ax.scatter(sol[dims_to_plot[0]], sol[dims_to_plot[1]], z_base,
                   color=colors[i % len(colors)], s=20, alpha=0.5)

        # Plot history only if show_history=True
        if show_history and history_dict and label in history_dict:
            hx, hy, hz = [], [], []
            for point in history_dict[label]:
                # Ensure point has correct dimension
                point = np.atleast_1d(point)
                if len(point) < len(bounds):
                    point = list(point) + extra_dims_vals[len(point):]
                hx.append(point[dims_to_plot[0]])
                hy.append(point[dims_to_plot[1]])
                full_point = [point[d] if d in dims_to_plot else extra_dims_vals[d] for d in range(len(bounds))]
                hz.append(func(full_point))
            # 3D trail
            ax.plot(hx, hy, hz, color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            # Projected trail on base
            ax.plot(hx, hy, [z_base]*len(hx), color=colors[i % len(colors)], linestyle=':', alpha=0.5)

    # Labels and legend
    ax.set_title(title)
    ax.set_xlabel(f"X{dims_to_plot[0]}")
    ax.set_ylabel(f"X{dims_to_plot[1]}")
    ax.set_zlabel("f(x)")
    ax.legend()
    
    # Return figure and axis instead of showing
    return fig, ax






# -----------------------------
# PyVista 3D Surface
# -----------------------------
def plot_function_surface_3d(func, bounds, results_dict, dims_to_plot, title="Function Surface",
                             grid_size=300, sphere_radius=0.2, clip_z=True, log_scale=False, extra_dims_vals=None):
    """
    3D surface plot for higher-dimensional functions.
    dims_to_plot: tuple/list of 2 integers specifying which dimensions to plot (e.g., [0,1])
    extra_dims_vals: list of values for other dimensions not plotted
    """

    x = np.linspace(bounds[dims_to_plot[0]][0], bounds[dims_to_plot[0]][1], grid_size)
    y = np.linspace(bounds[dims_to_plot[1]][0], bounds[dims_to_plot[1]][1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute Z values
    for i in range(grid_size):
        for j in range(grid_size):
            full_x = []
            for d in range(len(bounds)):
                if d == dims_to_plot[0]:
                    full_x.append(X[i,j])
                elif d == dims_to_plot[1]:
                    full_x.append(Y[i,j])
                else:
                    full_x.append(extra_dims_vals[d] if extra_dims_vals else bounds[d][1])
            Z[i,j] = func(full_x)

    # Optional clipping or log scale
    if clip_z:
        Z_max = np.percentile(Z,95)
        Z = np.clip(Z, np.min(Z), Z_max)
    if log_scale:
        Z = np.log1p(Z - np.min(Z))

    # PyVista structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap='viridis', opacity=0.9, show_edges=False, smooth_shading=True)
    plotter.enable_eye_dome_lighting()
    plotter.camera_position='iso'

    # Add axes for orientation
    plotter.add_axes(line_width=2)
    
    # Orientation legend text in viewport coordinates (bottom-left)
    #plotter.add_text("Orientation:\nX → Right\nY → Forward\nZ → Up", font_size=12, color='white', position='lower_left')

    colors = ['red','blue','green','orange','purple']
    z_base = np.min(Z)

    # Plot optimization points
    for i,(label, sol) in enumerate(results_dict.items()):
        full_sol = []
        for d in range(len(bounds)):
            if d in dims_to_plot:
                idx = dims_to_plot.index(d)
                full_sol.append(sol[d])
            else:
                full_sol.append(extra_dims_vals[d] if extra_dims_vals else bounds[d][1])
        z_val = func(full_sol)
        if clip_z:
            z_val = min(z_val,Z_max)
        if log_scale:
            z_val = np.log1p(z_val - np.min(Z))

        plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=[sol[dims_to_plot[0]], sol[dims_to_plot[1]], z_val]),
            color=colors[i%len(colors)]
        )
        plotter.add_point_labels(
            [np.array([sol[dims_to_plot[0]], sol[dims_to_plot[1]], z_val])],
            [f"{label}: {func(sol):.4f}"],
            font_size=12,
            point_color=colors[i%len(colors)]
        )
        plotter.add_mesh(
            pv.Sphere(radius=sphere_radius*0.5, center=[sol[dims_to_plot[0]], sol[dims_to_plot[1]], z_base]),
            color=colors[i%len(colors)],
            opacity=0.5
        )

    plotter.add_text(title, font_size=16)
    plotter.show()




# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(func, bounds, lr=0.01, max_iter=500, epsilon=1e-6, verbose=True, print_every=50):
    # Randomly initialize starting point x within the given bounds
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    
    history = []                     # Store function values at each iteration
    best_x = np.copy(x)              # Keep track of the best solution found
    best_val = func(x)               # Evaluate initial function value
    start_time = time.time()         # Record start time for runtime

    # -----------------------------
    # Numerical Gradient Calculation
    # -----------------------------
    def numerical_grad(f, x, h=1e-5):
        grad = np.zeros_like(x)      # Gradient vector initialization
        for i in range(len(x)):
            x1, x2 = np.copy(x), np.copy(x)
            x1[i] += h               # Perturb variable i positively
            x2[i] -= h               # Perturb variable i negatively
            # Central difference approximation of partial derivative
            grad[i] = (f(x1) - f(x2)) / (2 * h)
        return grad

    # -----------------------------
    # Main Gradient Descent Loop
    # -----------------------------
    for iteration in range(max_iter):
        grad = numerical_grad(func, x)        # Compute gradient at current x
        x_new = x - lr * grad                 # Update rule: move opposite to gradient
        step_size = np.linalg.norm(x_new - x) # Measure change in x (for convergence check)
        x = x_new                             
        current_val = func(x)                  # Evaluate function at new x
        history.append(current_val)            # Store function value

        # -----------------------------
        # Update Best Solution
        # -----------------------------
        if current_val < best_val:
            best_val = current_val
            best_x = np.copy(x)

        # Print progress periodically
        if verbose and (iteration % print_every == 0 or iteration == max_iter - 1):
            print(f"Iteration {iteration:4d} | f(x) = {current_val:.6f} | Step = {step_size:.6e}")

        # Convergence check: stop if step size is smaller than epsilon
        if step_size < epsilon:
            print(f"Converged after {iteration} iterations.")
            break

    # -----------------------------
    # Print total runtime and return results
    # -----------------------------
    total_time = time.time() - start_time
    print(f"GD runtime: {total_time:.3f} s\n")
    
    return best_x, best_val, history   # Return best solution, its value, and function history

# -----------------------------
# Genetic Algorithm
# -----------------------------
def genetic_algorithm(func, bounds, pop_size=6, generations=100, mutation_rate=0.3):
    dim = len(bounds)  # Number of variables/dimensions

    # -----------------------------
    # Initialize Population
    # -----------------------------
    # Each individual is randomly generated within the bounds
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    
    history = []             # Store best fitness of each generation
    start_time = time.time() # Record start time for runtime measurement

    # -----------------------------
    # Main Evolution Loop
    # -----------------------------
    for gen in range(generations):
        # Evaluate fitness of all individuals
        fitness = np.array([func(ind) for ind in pop])
        
        # Find the best individual of this generation
        best_idx = np.argmin(fitness)
        history.append(fitness[best_idx])  # Track best fitness
        
        # -----------------------------
        # Selection
        # -----------------------------
        # Select top 50% of individuals as parents for crossover
        parents = pop[np.argsort(fitness)[:pop_size//2]]

        # -----------------------------
        # Crossover (Recombination)
        # -----------------------------
        offspring = []
        for i in range(pop_size//2):
            # Randomly select two parents
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            if dim == 1:
                # For 1D, child is simple average of parents
                child = np.array([(p1[0] + p2[0]) / 2])
            else:
                # For multi-dimensional, pick a crossover point
                cp = np.random.randint(1, dim)
                # Combine first part from p1, second part from p2
                child = np.concatenate((p1[:cp], p2[cp:]))
            offspring.append(child)

        # -----------------------------
        # Mutation
        # -----------------------------
        offspring = np.array(offspring)
        for i, child in enumerate(offspring):
            # With mutation probability, randomly perturb one gene
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(dim)
                child[idx] += np.random.uniform(-1, 1)  # Small random change

        # -----------------------------
        # Form Next Generation
        # -----------------------------
        # Combine parents and offspring to form new population
        pop = np.vstack((parents, offspring))

    # -----------------------------
    # Return Best Solution
    # -----------------------------
    final_fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(final_fitness)
    best_ind = pop[best_idx]

    # Print runtime and best solution
    print(f"\nGA finished in {time.time() - start_time:.3f} s")
    print(f"Best GA solution: {best_ind}, f={func(best_ind):.6f}")
    
    return best_ind, func(best_ind), history  # Return best individual, its value, and fitness history

# -----------------------------
# Differential Evolution (Custom)
# -----------------------------
def differential_evolution_custom(func, bounds, pop_size=8, generations=200, F=0.8, CR=0.7):
    dim = len(bounds)  # Number of variables/dimensions

    # -----------------------------
    # Initialize Population
    # -----------------------------
    # Randomly generate initial population within bounds
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    
    # Evaluate initial fitness of each individual
    fitness = np.array([func(ind) for ind in pop])
    
    start_time = time.time()  # Record start time for runtime measurement

    # -----------------------------
    # Main Evolution Loop
    # -----------------------------
    for gen in range(generations):
        for i in range(pop_size):
            # -----------------------------
            # Mutation
            # -----------------------------
            # Choose 3 distinct individuals different from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # Generate mutant vector using differential mutation formula
            # v_i = a + F * (b - c)
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            # Clip to stay within variable bounds

            # -----------------------------
            # Crossover
            # -----------------------------
            # Decide which dimensions to take from mutant vector
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                # Ensure at least one dimension comes from mutant
                cross_points[np.random.randint(0, dim)] = True

            # Create trial vector by mixing mutant and current individual
            trial = np.where(cross_points, mutant, pop[i])

            # -----------------------------
            # Selection
            # -----------------------------
            # Evaluate trial vector fitness
            f_trial = func(trial)
            # Replace individual if trial is better
            if f_trial < fitness[i]:
                pop[i], fitness[i] = trial, f_trial

    # -----------------------------
    # Return Best Solution
    # -----------------------------
    best_idx = np.argmin(fitness)  # Index of best individual
    total_time = time.time() - start_time
    print(f"DE runtime: {total_time:.3f} s")
    print(f"Best DE solution: {pop[best_idx]}, f={fitness[best_idx]:.6f}")
    
    # Return best individual, its fitness, and fitness array of population
    return pop[best_idx], fitness[best_idx], fitness


import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Assume gradient_descent, genetic_algorithm, differential_evolution_custom, 
# plot_function_surface, and plot_function_surface_3d are already defined

class OptimizationGUI:
    def __init__(self, root):
        self.root = root    # root is the Tkinter main window.
        self.root.title("Function Optimization GUI")

        # -----------------------------
        # Top frame for all input group boxes
        # -----------------------------
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # -----------------------------
        # Function input group
        # -----------------------------
        func_frame = tk.LabelFrame(top_frame, text="Function Input", padx=10, pady=10)
        func_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Label(func_frame, text="Function (lambda):").grid(row=0, column=0, sticky="w")
        self.func_entry = tk.Entry(func_frame, width=40)
        self.func_entry.insert(0, "lambda x: x[0]**2 + x[0] + 5")
        self.func_entry.grid(row=0, column=1)

        tk.Label(func_frame, text="Bounds (tuple):").grid(row=1, column=0, sticky="w")
        self.bounds_entry = tk.Entry(func_frame, width=40)
        self.bounds_entry.insert(0, "(-5, 5)")
        self.bounds_entry.grid(row=1, column=1)

        tk.Label(func_frame, text="Dimensions to plot:").grid(row=2, column=0, sticky="w")
        self.dims_entry = tk.Entry(func_frame, width=20)
        self.dims_entry.insert(0, "0,1")
        self.dims_entry.grid(row=2, column=1)

        # -----------------------------
        # Clip Z radio
        # -----------------------------
        tk.Label(func_frame, text="Clip Z:").grid(row=3, column=0, sticky="w")
        self.clip_z_var = tk.BooleanVar(value=False)
        tk.Radiobutton(func_frame, text="False", variable=self.clip_z_var, value=False).grid(row=3, column=1, sticky="w")
        tk.Radiobutton(func_frame, text="True", variable=self.clip_z_var, value=True).grid(row=3, column=1, sticky="e")

        # -----------------------------
        # Log Scale radio
        # -----------------------------
        tk.Label(func_frame, text="Log Scale:").grid(row=4, column=0, sticky="w")
        self.log_scale_var = tk.BooleanVar(value=True)
        tk.Radiobutton(func_frame, text="False", variable=self.log_scale_var, value=False).grid(row=4, column=1, sticky="w")
        tk.Radiobutton(func_frame, text="True", variable=self.log_scale_var, value=True).grid(row=4, column=1, sticky="e")

        # -----------------------------
        # Gradient Descent parameters
        # -----------------------------
        gd_frame = tk.LabelFrame(top_frame, text="Gradient Descent", padx=10, pady=10)
        gd_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Label(gd_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w")
        self.gd_lr = tk.Entry(gd_frame, width=10)
        self.gd_lr.insert(0, "0.01")
        self.gd_lr.grid(row=0, column=1)

        tk.Label(gd_frame, text="Max Iterations:").grid(row=1, column=0, sticky="w")
        self.gd_max_iter = tk.Entry(gd_frame, width=10)
        self.gd_max_iter.insert(0, "500")
        self.gd_max_iter.grid(row=1, column=1)

        tk.Label(gd_frame, text="Epsilon:").grid(row=2, column=0, sticky="w")
        self.gd_epsilon = tk.Entry(gd_frame, width=10)
        self.gd_epsilon.insert(0, "1e-6")
        self.gd_epsilon.grid(row=2, column=1)

        # -----------------------------
        # Genetic Algorithm parameters
        # -----------------------------
        ga_frame = tk.LabelFrame(top_frame, text="Genetic Algorithm", padx=10, pady=10)
        ga_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Label(ga_frame, text="Population Size:").grid(row=0, column=0, sticky="w")
        self.ga_pop_size = tk.Entry(ga_frame, width=10)
        self.ga_pop_size.insert(0, "6")
        self.ga_pop_size.grid(row=0, column=1)

        tk.Label(ga_frame, text="Generations:").grid(row=1, column=0, sticky="w")
        self.ga_generations = tk.Entry(ga_frame, width=10)
        self.ga_generations.insert(0, "100")
        self.ga_generations.grid(row=1, column=1)

        tk.Label(ga_frame, text="Mutation Rate:").grid(row=2, column=0, sticky="w")
        self.ga_mutation_rate = tk.Entry(ga_frame, width=10)
        self.ga_mutation_rate.insert(0, "0.3")
        self.ga_mutation_rate.grid(row=2, column=1)

        # -----------------------------
        # Differential Evolution parameters
        # -----------------------------
        de_frame = tk.LabelFrame(top_frame, text="Differential Evolution", padx=10, pady=10)
        de_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Label(de_frame, text="Population Size:").grid(row=0, column=0, sticky="w")
        self.de_pop_size = tk.Entry(de_frame, width=10)
        self.de_pop_size.insert(0, "8")
        self.de_pop_size.grid(row=0, column=1)

        tk.Label(de_frame, text="Generations:").grid(row=1, column=0, sticky="w")
        self.de_generations = tk.Entry(de_frame, width=10)
        self.de_generations.insert(0, "200")
        self.de_generations.grid(row=1, column=1)

        tk.Label(de_frame, text="F:").grid(row=2, column=0, sticky="w")
        self.de_F = tk.Entry(de_frame, width=10)
        self.de_F.insert(0, "0.8")
        self.de_F.grid(row=2, column=1)

        tk.Label(de_frame, text="CR:").grid(row=3, column=0, sticky="w")
        self.de_CR = tk.Entry(de_frame, width=10)
        self.de_CR.insert(0, "0.7")
        self.de_CR.grid(row=3, column=1)

        # -----------------------------
        # Run buttons
        # -----------------------------
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.run_button = tk.Button(button_frame, text="Run Optimization", command=self.run_optimization)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.pv_button = tk.Button(button_frame, text="Show PyVista 3D Plot", command=self.run_pyvista_plot)
        self.pv_button.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # Results group box (below inputs)
        # -----------------------------
        self.results_frame = tk.LabelFrame(root, text="Best Values", padx=10, pady=10)
        self.results_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.gd_label = tk.Label(self.results_frame, text="Gradient Descent: N/A", fg="red")
        self.gd_label.pack(anchor="w")
        self.ga_label = tk.Label(self.results_frame, text="Genetic Algorithm: N/A", fg="green")
        self.ga_label.pack(anchor="w")
        self.de_label = tk.Label(self.results_frame, text="Differential Evolution: N/A", fg="blue")
        self.de_label.pack(anchor="w")

        # -----------------------------
        # Plot frame
        # -----------------------------
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_plot = tk.Frame(self.plot_frame)
        self.left_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_plot = tk.Frame(self.plot_frame)
        self.right_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Store variables
        self.func = None
        self.bounds = None
        self.results_dict = None
        self.history_dict = None
        self.dims_to_plot = None
        self.extra_dims_vals = None

    # run_optimization and run_pyvista_plot remain unchanged except pass:
    # clip_z=self.clip_z_var.get(), log_scale=self.log_scale_var.get()


    def run_optimization(self):
        # ----- Parse function -----
        try:
            self.func = eval(self.func_entry.get())
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {e}")
            return

        # ----- Parse bounds -----
        try:
            bounds = eval(self.bounds_entry.get())
            if isinstance(bounds[0], (int, float)):
                bounds = [bounds]
            self.bounds = bounds
        except Exception as e:
            messagebox.showerror("Error", f"Invalid bounds: {e}")
            return

        dim = len(self.bounds)

        # ----- Parse dimensions to plot -----
        if dim > 1:
            try:
                self.dims_to_plot = [int(i.strip()) for i in self.dims_entry.get().split(",")]
                if len(self.dims_to_plot) != 2 or max(self.dims_to_plot) >= dim:
                    raise ValueError
            except:
                messagebox.showerror("Error", "Invalid dimensions to plot")
                return
        else:
            self.dims_to_plot = [0]

        # ----- Extra dims values (upper bounds) -----
        self.extra_dims_vals = [b[1] for b in self.bounds]

        # ----- Read optimizer parameters -----
        gd_lr = float(self.gd_lr.get())
        gd_max_iter = int(self.gd_max_iter.get())
        gd_epsilon = float(self.gd_epsilon.get())

        ga_pop_size = int(self.ga_pop_size.get())
        ga_generations = int(self.ga_generations.get())
        ga_mutation_rate = float(self.ga_mutation_rate.get())

        de_pop_size = int(self.de_pop_size.get())
        de_generations = int(self.de_generations.get())
        de_F = float(self.de_F.get())
        de_CR = float(self.de_CR.get())

        # ----- Run optimizers with user parameters -----
        gd_sol, gd_val, gd_hist = gradient_descent(self.func, self.bounds,
                                                   lr=gd_lr, max_iter=gd_max_iter, epsilon=gd_epsilon)
        ga_sol, ga_val, ga_hist = genetic_algorithm(self.func, self.bounds,
                                                    pop_size=ga_pop_size, generations=ga_generations,
                                                    mutation_rate=ga_mutation_rate)
        de_sol, de_val, de_hist = differential_evolution_custom(self.func, self.bounds,
                                                                pop_size=de_pop_size, generations=de_generations,
                                                                F=de_F, CR=de_CR)

        self.results_dict = {'GD': gd_sol, 'GA': ga_sol, 'DE': de_sol}
        self.history_dict = {'GD': gd_hist, 'GA': ga_hist, 'DE': de_hist}

        # ----- Update results group -----
        self.gd_label.config(text=f"Gradient Descent: {gd_val:.4f} at {gd_sol}")
        self.ga_label.config(text=f"Genetic Algorithm: {ga_val:.4f} at {ga_sol}")
        self.de_label.config(text=f"Differential Evolution: {de_val:.4f} at {de_sol}")

        # ----- Clear previous plots -----
        for widget in self.left_plot.winfo_children():
            widget.destroy()
        for widget in self.right_plot.winfo_children():
            widget.destroy()

        # ----- Left plot: convergence -----
        fig1, ax1 = plt.subplots(figsize=(5,4))
        ax1.plot(gd_hist, label=f'GD (min={gd_val:.4f})', color='red')
        ax1.plot(ga_hist, label=f'GA (min={ga_val:.4f})', color='green')
        ax1.plot(de_hist, label=f'DE (min={de_val:.4f})', color='blue')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("f(x)")
        ax1.set_title("Convergence Curves")
        ax1.legend()
        canvas1 = FigureCanvasTkAgg(fig1, master=self.left_plot)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ----- Right plot: function -----
        if dim == 1:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            x_vals = np.linspace(self.bounds[0][0], self.bounds[0][1], 400)
            y_vals = np.array([self.func([x]) for x in x_vals])
            ax2.plot(x_vals, y_vals, label="f(x)")
            ax2.scatter(gd_sol[0], gd_val, color='red', label=f'GD (min={gd_val:.4f})')
            ax2.scatter(ga_sol[0], ga_val, color='green', label=f'GA (min={ga_val:.4f})')
            ax2.scatter(de_sol[0], de_val, color='blue', label=f'DE (min={de_val:.4f})')
            ax2.set_xlabel("x")
            ax2.set_ylabel("f(x)")
            ax2.set_title("1D Function Optimization")
            ax2.legend()
        else:
            # For 2D+ functions, get figure and axis from plotting function
            fig2, ax2 = plot_function_surface(
                self.func,
                self.bounds,
                self.results_dict,
                dims_to_plot=self.dims_to_plot,
                title="3D Function Optimization",
                clip_z=self.clip_z_var.get(),
                log_scale=self.log_scale_var.get(),
                extra_dims_vals=self.extra_dims_vals,
                grid_size=200,
                history_dict=self.history_dict
            )

        canvas2 = FigureCanvasTkAgg(fig2, master=self.right_plot)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_pyvista_plot(self):
        # ----- Parse bounds -----
        try:
            bounds = eval(self.bounds_entry.get())
            if isinstance(bounds[0], (int, float)):
                bounds = [bounds]
            self.bounds = bounds
        except Exception as e:
            messagebox.showerror("Error", f"Invalid bounds: {e}")
            return
        
        dim = len(self.bounds)
        
        # ----- Parse dimensions to plot -----
        if dim > 1:
            if not all([self.func, self.bounds, self.history_dict]):
                messagebox.showwarning("Warning", "Run optimization first!")
                return
            
            plot_function_surface_3d(
                func=self.func,
                bounds=self.bounds,
                results_dict=self.results_dict,
                dims_to_plot=self.dims_to_plot,
                title="3D Function Optimization",
                grid_size=300,
                sphere_radius=0.2,
                clip_z=self.clip_z_var.get(),
                log_scale=self.log_scale_var.get(),
                extra_dims_vals=self.extra_dims_vals
            )
        else:
            return messagebox.showerror("Error", "Dimension must be more than 2")


if __name__ == "__main__":
    root = tk.Tk()
    gui = OptimizationGUI(root)
    root.geometry("1200x600")
    root.mainloop()

