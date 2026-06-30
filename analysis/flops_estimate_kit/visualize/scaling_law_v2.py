import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from .scaling_law import get_rl_scatters

import warnings
warnings.filterwarnings('ignore')

class AdvancedFLOPsPerformanceFitter_v2:
    def __init__(self, data, default_config=None, psft_by_benchmark=None):
        """
        Initialize the advanced fitter.
        
        Parameters:
        -----------
        data : dict
            Performance data
        default_config : dict
            Default config applied to all benchmarks without per-benchmark overrides
        psft_by_benchmark : dict or None
            Raw performance at each benchmark at the RL start (SFT branch), shown as Psft in subplot info boxes
        """
        self.data = data
        self.psft_by_benchmark = psft_by_benchmark or {}
        self.flops = np.array(list(data.keys()))
        self.all_benchmarks = list(next(iter(data.values())).keys())
        
        # Default config - includes fixed_C0 parameter
        self.default_config = default_config or {
            'split_mode': 'percentage',  # 'percentage', 'compute', 'index'
            'train_range': (20, 60),
            'val_range': (60, 80),
            'test_range': (80, 100),  # test set range
            'use_test_set': False,  # whether to use a test set
            'detect_outliers': True,
            'outlier_threshold': 2.5,
            'use_robust_regression': False,
            'lts_alpha': 0.75,
            'model_types': 'auto',
            'metric': 'val_rmse',
            'logistic_optimization': 'adaptive',
            'fixed_C0': None  # None = fit C0 automatically; set a value to fix C0
        }
        
        # Per-benchmark config overrides
        self.benchmark_configs = {}
        
        # All available model functions
        self.available_models = {
            'linear': lambda x, a, b: a * x + b,
            'log': lambda x, a, b, c: a * np.log(x + c) + b,
            'power': lambda x, a, b, c: a * np.power(x + c, b),
            'exponential': lambda x, a, b, c: a * (1 - np.exp(-b * x)) + c,
            'sigmoid': lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) + d,
            'logistic': self.logistic_curve
        }
    
    def set_benchmark_config(self, benchmark_name, config):
        """
        Set config for one or more benchmarks.
        
        Parameters:
        -----------
        benchmark_name : str or list
            Benchmark name or list of names
        config : dict
            Config for the benchmark(s)
        """
        if isinstance(benchmark_name, str):
            benchmark_names = [benchmark_name]
        else:
            benchmark_names = benchmark_name
            
        for name in benchmark_names:
            if name not in self.all_benchmarks:
                print(f"Warning: Benchmark '{name}' not found. Available: {self.all_benchmarks}")
                continue
            
            # Merge default config with benchmark-specific overrides
            merged_config = self.default_config.copy()
            merged_config.update(config)
            self.benchmark_configs[name] = merged_config
            print(f"Config set for benchmark '{name}': {merged_config}")
    
    def set_benchmark_configs_batch(self, configs_dict):
        """
        Set configs for multiple benchmarks in batch.
        
        Parameters:
        -----------
        configs_dict : dict
            Format: {benchmark_name: config_dict, ...}
        """
        for benchmark_name, config in configs_dict.items():
            self.set_benchmark_config(benchmark_name, config)
    
    def get_benchmark_config(self, benchmark_name):
        """
        Get config for a specific benchmark.
        
        Parameters:
        -----------
        benchmark_name : str
            Benchmark name
        
        Returns:
        --------
        config : dict
            Config for the benchmark
        """
        if benchmark_name in self.benchmark_configs:
            return self.benchmark_configs[benchmark_name]
        return self.default_config.copy()
    
    def select_benchmarks(self, benchmark_filter=None):
        """
        Select benchmarks to analyze.
        
        Parameters:
        -----------
        benchmark_filter : None, str, list, or callable
            - None: use all benchmarks
            - str: benchmarks whose name contains this string
            - list: explicit list of benchmark names
            - callable: custom filter function
        
        Returns:
        --------
        selected_benchmarks : list
            Selected benchmark names
        """
        if benchmark_filter is None:
            return self.all_benchmarks
        
        elif isinstance(benchmark_filter, str):
            # Substring match
            return [b for b in self.all_benchmarks if benchmark_filter.lower() in b.lower()]
        
        elif isinstance(benchmark_filter, list):
            # Explicit list
            valid_benchmarks = []
            for b in benchmark_filter:
                if b in self.all_benchmarks:
                    valid_benchmarks.append(b)
                else:
                    print(f"Warning: Benchmark '{b}' not found")
            return valid_benchmarks
        
        elif callable(benchmark_filter):
            # Custom filter function
            return [b for b in self.all_benchmarks if benchmark_filter(b)]
        
        else:
            print(f"Invalid benchmark_filter type: {type(benchmark_filter)}")
            return self.all_benchmarks
    
    def get_benchmark_data(self, benchmark_name):
        """Get data for the specified benchmark."""
        if benchmark_name not in self.all_benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found. Available: {self.all_benchmarks}")
        
        performance = np.array([self.data[flop][benchmark_name] for flop in self.flops])
        return self.flops, performance
    
    def logistic_curve(self, x, *params):
        """
        Logistic curve function with optional fixed C0.
        When C0 is fixed, params are (B, Cmid, A).
        When C0 is not fixed, params are (B, C0, Cmid, A).
        """
        # Read current config to determine whether C0 is fixed
        # Note: config is passed via instance context
        if hasattr(self, '_current_config') and self._current_config.get('fixed_C0') is not None:
            C0 = self._current_config['fixed_C0']
            if len(params) == 3:
                B, Cmid, A = params
            else:
                # Backward-compatible older format
                B, _, Cmid, A = params
                C0 = params[1]
        else:
            if len(params) == 4:
                B, C0, Cmid, A = params
            else:
                # Default behavior
                raise ValueError(f"Invalid number of parameters for logistic curve: {len(params)}")
        
        # Stable form: y = C0 + (A-C0)/(1+(Cmid/x)^B) = C0 + (A-C0)*expit(-B*log(Cmid/(x+eps)))
        # Avoid overflow when B is large and x is small (clip(ratio)**B), which can NaN the whole fit curve
        x = np.asarray(x, dtype=float)
        eps = 1e-10
        log_ratio = np.log(np.maximum(Cmid, eps)) - np.log(x + eps)
        z = B * log_ratio
        with np.errstate(over="ignore", invalid="ignore"):
            return C0 + (A - C0) * expit(-z)
    
    def lts_regression(self, x, y, model_func, initial_params=None, max_iter=100, lts_alpha=0.75):
        """Least Trimmed Squares (LTS) robust regression"""
        n = len(x)
        h = int(n * lts_alpha)
        
        if h < len(initial_params) if initial_params is not None else 2:
            h = n
        
        best_params = initial_params
        best_loss = np.inf
        best_inliers = np.arange(n)
        
        if initial_params is None:
            try:
                initial_params, _ = curve_fit(model_func, x, y, maxfev=5000)
            except:
                return None, None
        
        rng = np.random.default_rng(42)
        current_params = initial_params.copy()
        
        for iteration in range(max_iter):
            try:
                y_pred = model_func(x, *current_params)
                residuals = (y - y_pred) ** 2
                
                sorted_indices = np.argsort(residuals)
                selected_indices = sorted_indices[:h]
                
                x_selected = x[selected_indices]
                y_selected = y[selected_indices]
                
                new_params, _ = curve_fit(model_func, x_selected, y_selected, 
                                         p0=current_params, maxfev=5000)
                
                y_pred_new = model_func(x, *new_params)
                residuals_new = (y - y_pred_new) ** 2
                sorted_residuals_new = np.sort(residuals_new)
                current_loss = np.sum(sorted_residuals_new[:h])
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = new_params.copy()
                    best_inliers = selected_indices.copy()
                    current_params = new_params.copy()
                else:
                    if iteration < max_iter - 1:
                        current_params = best_params + rng.normal(size=len(best_params)) * 0.01
                
            except Exception:
                if iteration < max_iter - 1:
                    current_params = best_params + rng.normal(size=len(best_params)) * 0.01
                continue
        
        return best_params, best_inliers
    
    def fit_model_robust(self, x_train, y_train, model_name, model_func, config):
        """Fit a model using robust regression."""
        y_min, y_max = np.min(y_train), np.max(y_train)
        x_min, x_max = np.min(x_train), np.max(x_train)
        # Set current config context for logistic_curve
        self._current_config = config
        
        # Initial parameter guesses
        if model_name == 'linear':
            initial_params = [1, y_train[0]]
        elif model_name == 'log':
            initial_params = [10, y_train[0], 1]
        elif model_name == 'power':
            initial_params = [y_max, 0.5, 1]
        elif model_name == 'exponential':
            initial_params = [y_max - y_min, 0.01, y_min]
        elif model_name == 'sigmoid':
            initial_params = [y_max - y_min, 0.01, np.mean(x_train), y_min]
        elif model_name == 'logistic':
            # Initial params depend on whether C0 is fixed
            if config.get('fixed_C0') is not None:
                # Fixed C0: only 3 params (B, Cmid, A)
                initial_params = [1.0, np.median(x_train), y_max]
            else:
                # Free C0: 4 params (B, C0, Cmid, A)
                initial_params = [1.0, y_min, np.median(x_train), y_max]
        else:
            initial_params = None
        
        if config['use_robust_regression']:
            params, inliers = self.lts_regression(
                x_train, y_train, model_func, 
                initial_params, max_iter=50, 
                lts_alpha=config['lts_alpha']
            )
            if params is None:
                try:
                    params, _ = curve_fit(model_func, x_train, y_train, 
                                        p0=initial_params, maxfev=10000)
                    inliers = np.arange(len(x_train))
                except:
                    return None, None
        else:
            try:
                if model_name == 'logistic':
                    params = self.fit_logistic_scipy(x_train, y_train)
                    print('use fit_logistic_grid_search:', (params is None))
                    if params is None:
                        params = self.fit_logistic_grid_search(
                            x_train, y_train, config['logistic_optimization']
                        )
                else:
                    params, _ = curve_fit(model_func, x_train, y_train, 
                                        p0=initial_params, maxfev=10000)
                inliers = np.arange(len(x_train))
            except:
                return None, None
        
        return params, inliers
    
    def fit_logistic_grid_search(self, x_train, y_train, optimization_mode='comprehensive'):
        """Fit logistic curve via grid search."""
        best_params = None
        best_rmse = np.inf
        
        y_min, y_max = np.min(y_train), np.max(y_train)
        x_min, x_max = np.min(x_train), np.max(x_train)
        
        # Read current config
        config = self._current_config if hasattr(self, '_current_config') else {}
        fixed_C0 = config.get('fixed_C0')
        print('check optimization_mode:', optimization_mode)
        if optimization_mode == 'comprehensive':
            B_range = np.logspace(-1, 2, 15)
            Cmid_range = np.linspace(x_min, x_max, 15)
            A_range = np.linspace(y_max - 0.3 * (y_max - y_min), y_max + 0.1 * (y_max - y_min), 10)
            
            if fixed_C0 is not None:
                # Fixed C0: search only B, Cmid, A
                for B in B_range:
                    for Cmid in Cmid_range:
                        for A in A_range:
                            try:
                                y_pred = self.logistic_curve(x_train, B, Cmid, A)
                                if not np.any(np.isnan(y_pred)):
                                    rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
                                    if rmse < best_rmse:
                                        best_rmse = rmse
                                        best_params = [B, Cmid, A]
                            except:
                                continue
            else:
                # Free C0: search all 4 parameters
                C0_range = np.linspace(y_min - 0.1 * (y_max - y_min), y_min + 0.3 * (y_max - y_min), 10)
                for B in B_range:
                    for C0 in C0_range:
                        for Cmid in Cmid_range:
                            for A in A_range:
                                try:
                                    y_pred = self.logistic_curve(x_train, B, C0, Cmid, A)
                                    if not np.any(np.isnan(y_pred)):
                                        rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
                                        if rmse < best_rmse:
                                            best_rmse = rmse
                                            best_params = [B, C0, Cmid, A]
                                except:
                                    continue
        
        elif optimization_mode == 'adaptive':
            def objective(params):
                try:
                    y_pred = self.logistic_curve(x_train, *params)
                    if np.any(np.isnan(y_pred)):
                        return 1e10
                    return np.sqrt(np.mean((y_train - y_pred) ** 2))
                except:
                    return 1e10
            
            if fixed_C0 is not None:
                # Bounds when C0 is fixed
                bounds = [
                    (0.1, 100),  # B
                    (x_min * 0.5, x_max * 1.5),  # Cmid
                    (y_max - 0.5 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))  # A
                ]
            else:
                # Bounds when C0 is free
                bounds = [
                    (0.1, 100),  # B
                    (y_min - 0.2 * (y_max - y_min), y_min + 0.5 * (y_max - y_min)),  # C0
                    (x_min * 0.5, x_max * 1.5),  # Cmid
                    (y_max - 0.5 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))  # A
                ]
            
            result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
            if result.success:
                best_params = result.x
                best_rmse = result.fun
        
        return best_params
    
    def fit_logistic_scipy(self, x_train, y_train):
        """Fit logistic curve with scipy curve_fit."""
        y_min, y_max = np.min(y_train), np.max(y_train)
        x_min, x_max = np.min(x_train), np.max(x_train)
        x_mid = np.median(x_train)
        
        # Read current config
        config = self._current_config if hasattr(self, '_current_config') else {}
        fixed_C0 = config.get('fixed_C0')
        
        if fixed_C0 is not None:
            # Wrapper with C0 fixed
            def logistic_fixed_C0(x, B, Cmid, A):
                return self.logistic_curve(x, B, Cmid, A)
            
            initial_guess = [1.0, x_mid, y_max]
            bounds = (
                [0.01, x_min * 0.1, y_max - 0.5 * (y_max - y_min)],
                [100, x_max * 2, y_max + 0.5 * (y_max - y_min)]
            )
            
            try:
                popt, _ = curve_fit(logistic_fixed_C0, x_train, y_train, 
                                p0=initial_guess, bounds=bounds, maxfev=10000)
                return popt
            except:
                return None
        else:
            # Original 4-parameter fit
            initial_guess = [1.0, y_min, x_mid, y_max]
            bounds = (
                [0.01, y_min - 0.5 * (y_max - y_min), x_min * 0.1, y_max - 0.5 * (y_max - y_min)],
                [100, y_min + 0.5 * (y_max - y_min), x_max * 2, y_max + 0.5 * (y_max - y_min)]
            )
            
            try:
                popt, _ = curve_fit(self.logistic_curve, x_train, y_train, 
                                p0=initial_guess, bounds=bounds, maxfev=10000)
                return popt
            except:
                return None
    
    def detect_outliers_iterative(self, x, y, model_func, config, max_iterations=3):
        """Iteratively detect outliers."""
        if not config['detect_outliers']:
            return np.zeros(len(x), dtype=bool)
        
        x_clean, y_clean = x.copy(), y.copy()
        outlier_mask = np.zeros(len(x), dtype=bool)
        
        for iteration in range(max_iterations):
            try:
                model_name = [k for k, v in self.available_models.items() if v == model_func][0]
                
                popt, _ = self.fit_model_robust(x_clean, y_clean, model_name, model_func, config)
                if popt is None:
                    break
                
                y_pred_all = model_func(x, *popt)
                residuals = y - y_pred_all
                
                mad = np.median(np.abs(residuals - np.median(residuals)))
                modified_z_scores = 0.6745 * (residuals - np.median(residuals)) / (mad if mad > 0 else 1)
                
                new_outlier_mask = np.abs(modified_z_scores) > config['outlier_threshold']
                
                if np.array_equal(outlier_mask, new_outlier_mask):
                    break
                
                outlier_mask = new_outlier_mask
                x_clean = x[~outlier_mask]
                y_clean = y[~outlier_mask]
                
            except Exception:
                break
        
        return outlier_mask
    
    def split_data_by_range(self, x, y, config):
        """
        Split data according to the configured mode.
        
        Supported modes:
        1. percentage: by fraction of data points (0-100)
        2. compute: by compute budget (petaFLOPs)
        3. index: by absolute index
        """
        n_total = len(x)
        split_mode = config.get('split_mode', 'percentage')
        train_range = config['train_range']
        val_range = config['val_range']
        use_test_set = config.get('use_test_set', False)
        test_range = config.get('test_range', None) if use_test_set else None
        
        # Initialize masks
        train_mask = np.zeros(n_total, dtype=bool)
        val_mask = np.zeros(n_total, dtype=bool)
        test_mask = np.zeros(n_total, dtype=bool)
        ignored_mask = np.ones(n_total, dtype=bool)
        
        if split_mode == 'percentage':
            # Percentage-based split
            train_start_idx = int(n_total * train_range[0] / 100)
            train_end_idx = int(n_total * train_range[1] / 100)
            val_start_idx = int(n_total * val_range[0] / 100)
            val_end_idx = int(n_total * val_range[1] / 100)
            
            train_mask[train_start_idx:train_end_idx] = True
            val_mask[val_start_idx:val_end_idx] = True
            
            if use_test_set and test_range:
                test_start_idx = int(n_total * test_range[0] / 100)
                test_end_idx = int(n_total * test_range[1] / 100)
                test_mask[test_start_idx:test_end_idx] = True
        
        elif split_mode == 'compute':
            # Compute-based split (petaFLOPs); x should be petaFLOPs
            train_mask = (x >= train_range[0]) & (x < train_range[1])
            val_mask = (x >= val_range[0]) & (x < val_range[1])
            
            if use_test_set and test_range:
                test_mask = (x >= test_range[0]) & (x < test_range[1])
        
        elif split_mode == 'index':
            # Absolute index split
            train_start, train_end = train_range
            val_start, val_end = val_range
            
            # Clamp indices to valid range
            train_start = max(0, min(train_start, n_total))
            train_end = max(0, min(train_end, n_total))
            val_start = max(0, min(val_start, n_total))
            val_end = max(0, min(val_end, n_total))
            
            train_mask[train_start:train_end] = True
            val_mask[val_start:val_end] = True
            
            if use_test_set and test_range:
                test_start, test_end = test_range
                test_start = max(0, min(test_start, n_total))
                test_end = max(0, min(test_end, n_total))
                test_mask[test_start:test_end] = True
        
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}. Use 'percentage', 'compute', or 'index'.")
        
        # Update ignored_mask
        ignored_mask[train_mask | val_mask | test_mask] = False
        
        return train_mask, val_mask, test_mask, ignored_mask
    
    def format_equation(self, model_name, params):
        """Format equation string."""
        if model_name == 'linear':
            a, b = params
            return f"y = {a:.3e}x + {b:.3f}"
        elif model_name == 'log':
            a, b, c = params
            return f"y = {a:.3f}·ln(x + {c:.3f}) + {b:.3f}"
        elif model_name == 'power':
            a, b, c = params
            return f"y = {a:.3f}·(x + {c:.3f})^{b:.3f}"
        elif model_name == 'exponential':
            a, b, c = params
            return f"y = {a:.3f}·(1 - e^(-{b:.3e}x)) + {c:.3f}"
        elif model_name == 'sigmoid':
            a, b, c, d = params
            return f"y = {a:.3f}/(1 + e^(-{b:.3e}(x - {c:.1f}))) + {d:.3f}"
        elif model_name == 'logistic':
            # Check whether current config fixes C0
            config = self._current_config if hasattr(self, '_current_config') else {}
            fixed_C0 = config.get('fixed_C0')
            
            if fixed_C0 is not None and len(params) == 3:
                B, Cmid, A = params
                C0 = fixed_C0
                return f"y = {C0:.2f} + {A-C0:.2f}/(1+({Cmid:.0f}/x)^{B:.2f}) [C0 fixed]"
            else:
                B, C0, Cmid, A = params
                return f"y = {C0:.2f} + {A-C0:.2f}/(1+({Cmid:.0f}/x)^{B:.2f})"
        else:
            return f"{model_name}"
    
    def select_model_types(self, model_types):
        """Select model types to fit."""
        if model_types == 'auto':
            return self.available_models.copy()
        elif model_types == 'linear':
            return {'linear': self.available_models['linear']}
        elif model_types == 'nonlinear':
            return {k: v for k, v in self.available_models.items() if k != 'linear'}
        elif isinstance(model_types, list):
            selected_models = {}
            for model_name in model_types:
                if model_name in self.available_models:
                    selected_models[model_name] = self.available_models[model_name]
                else:
                    print(f"Warning: Model '{model_name}' not found")
            return selected_models
        else:
            print(f"Invalid model_types: {model_types}. Using 'auto'.")
            return self.available_models.copy()
    
    def fit_curves(self, benchmark_name):
        """Fit multiple curve models for the specified benchmark."""
        # Benchmark-specific config
        config = self.get_benchmark_config(benchmark_name)
        # Set config context for logistic_curve / format_equation
        self._current_config = config
        
        # Model selection
        models = self.select_model_types(config['model_types'])
        
        if not models:
            print(f"No models selected for benchmark {benchmark_name}")
            return {}
        
        x, y = self.get_benchmark_data(benchmark_name)
        
        # Split data using configured ranges
        train_mask, val_mask, test_mask, ignored_mask = self.split_data_by_range(x, y, config)

        # Optional: drop y<0 from val (negative delta stays in ignored, excluded from val metrics)
        if config.get('exclude_nonpositive_val', False):
            val_mask = val_mask & (y >= 0)
            ignored_mask = np.ones(len(x), dtype=bool)
            ignored_mask[train_mask | val_mask | test_mask] = False
        
        results = {}
        
        for model_name, model_func in models.items():
            try:
                x_train_raw = x[train_mask]
                y_train_raw = y[train_mask]
                
                outlier_mask_train = self.detect_outliers_iterative(
                    x_train_raw, y_train_raw, model_func, config
                )
                
                outlier_mask_global = np.zeros(len(x), dtype=bool)
                outlier_mask_global[train_mask] = outlier_mask_train
                
                x_train = x_train_raw[~outlier_mask_train]
                y_train = y_train_raw[~outlier_mask_train]
                
                x_val = x[val_mask]
                y_val = y[val_mask]
                
                x_test = x[test_mask]
                y_test = y[test_mask]
                
                if len(x_train) < 2:
                    continue
                
                popt, inliers = self.fit_model_robust(x_train, y_train, model_name, model_func, config)
                
                if popt is None:
                    continue
                
                if config['use_robust_regression'] and inliers is not None:
                    x_train_robust = x_train[inliers]
                    y_train_robust = y_train[inliers]
                    
                    robust_mask = np.zeros(len(x), dtype=bool)
                    train_indices = np.where(train_mask & ~outlier_mask_global)[0]
                    robust_mask[train_indices[inliers]] = True
                else:
                    x_train_robust = x_train
                    y_train_robust = y_train
                    robust_mask = train_mask & ~outlier_mask_global
                
                # Train metrics
                y_pred_train = model_func(x_train_robust, *popt)
                r2_train = r2_score(y_train_robust, y_pred_train)
                rmse_train = np.sqrt(np.mean((y_train_robust - y_pred_train)**2))
                
                # Validation metrics
                if len(x_val) > 0:
                    y_pred_val = model_func(x_val, *popt)
                    r2_val = r2_score(y_val, y_pred_val)
                    rmse_val = np.sqrt(np.mean((y_val - y_pred_val)**2))
                else:
                    r2_val = r2_train
                    rmse_val = rmse_train
                
                # Test metrics (when enabled)
                if config.get('use_test_set', False) and len(x_test) > 0:
                    y_pred_test = model_func(x_test, *popt)
                    r2_test = r2_score(y_test, y_pred_test)
                    rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
                else:
                    r2_test = None
                    rmse_test = None
                
                n = len(x_train_robust)
                k = len(popt)
                rss = np.sum((y_train_robust - y_pred_train)**2)
                
                aic = n * np.log(rss/n) + 2 * k if n > 0 else np.inf
                bic = n * np.log(rss/n) + k * np.log(n) if n > 0 else np.inf
                
                results[model_name] = {
                    'params': popt,
                    'r2_train': r2_train,
                    'rmse_train': rmse_train,
                    'r2_val': r2_val,
                    'rmse_val': rmse_val,
                    'r2_test': r2_test,
                    'rmse_test': rmse_test,
                    'aic': aic,
                    'bic': bic,
                    'func': model_func,
                    'x_all': x,
                    'y_all': y,
                    'train_mask': train_mask,
                    'val_mask': val_mask,
                    'test_mask': test_mask,
                    'ignored_mask': ignored_mask,
                    'outlier_mask': outlier_mask_global,
                    'robust_mask': robust_mask if config['use_robust_regression'] else None,
                    'equation': self.format_equation(model_name, popt),
                    'n_params': k,
                    'config': config
                }
                
            except Exception as e:
                print(f"Error fitting {model_name} for {benchmark_name}: {e}")
                continue
                
        return results
    
    def get_best_model(self, benchmark_name):
        """Return the best-fitting model for a benchmark."""
        config = self.get_benchmark_config(benchmark_name)
        metric = config['metric']
        
        results = self.fit_curves(benchmark_name)
        if not results:
            return None, None
        
        if metric == 'auto':
            scores = {}
            for model_name, result in results.items():
                val_rmse_score = -result['rmse_val']
                val_r2_score = result['r2_val']
                aic_score = -result['aic']
                bic_score = -result['bic']
                
                total_score = (
                    0.4 * (val_rmse_score / abs(val_rmse_score) if val_rmse_score != 0 else 0) +
                    0.3 * val_r2_score +
                    0.15 * (aic_score / abs(aic_score) if aic_score != 0 else 0) +
                    0.15 * (bic_score / abs(bic_score) if bic_score != 0 else 0)
                )
                scores[model_name] = total_score
            
            best_model_name = max(scores, key=scores.get)
            return best_model_name, results[best_model_name]
        
        elif metric == 'val_rmse':
            best_model = min(results.items(), key=lambda x: x[1]['rmse_val'])
        elif metric == 'val_r2':
            best_model = max(results.items(), key=lambda x: x[1]['r2_val'])
        elif metric == 'aic':
            best_model = min(results.items(), key=lambda x: x[1]['aic'])
        elif metric == 'bic':
            best_model = min(results.items(), key=lambda x: x[1]['bic'])
        else:
            print(f"Unknown metric: {metric}. Using 'val_rmse'.")
            best_model = min(results.items(), key=lambda x: x[1]['rmse_val'])
        
        return best_model[0], best_model[1]
    
    def convert_to_exaflops(self, petaflops):
        """Convert PetaFLOPs to ExaFLOPs."""
        return petaflops / 1
    
    def format_range_for_display(self, range_tuple, split_mode):
        """Format a range tuple for display."""
        if split_mode == 'percentage':
            return f"{range_tuple[0]}-{range_tuple[1]}%"
        elif split_mode == 'compute':
            return f"{range_tuple[0]:.0f}-{range_tuple[1]:.0f}P"
        elif split_mode == 'index':
            return f"[{range_tuple[0]}:{range_tuple[1]}]"
        else:
            return str(range_tuple)
    
    def plot_selected_benchmarks(
            self, 
            sft_ckpt, 
            benchmark_filter=None, 
            figsize=None,
            save_path=None,
            dpi=300,
            save_format='png',
            title_dataset='SFT',
            ):
        """Plot best-fit curves for selected benchmarks."""
        
        # Select benchmarks
        selected_benchmarks = self.select_benchmarks(benchmark_filter)
        
        if not selected_benchmarks:
            print("No benchmarks selected for plotting")
            return {}
        
        # Dynamic figure size
        n_benchmarks = len(selected_benchmarks)
        if figsize is None:
            n_cols = min(4, n_benchmarks)
            n_rows = int(np.ceil(n_benchmarks / n_cols))
            if n_benchmarks == 1:
                figsize = (7.2, 5.6)
            else:
                figsize = (5 * n_cols, 4 * n_rows)
        else:
            n_cols = 4
            n_rows = int(np.ceil(n_benchmarks / n_cols))
        
        # Subplots
        fig = plt.figure(figsize=figsize)
        
        # Title
        title_str = f'RL scaling for {title_dataset} CKPT-{sft_ckpt}'
        
        fig.suptitle(title_str, fontsize=18, y=0.975)
        
        # Accumulated results
        all_results = {}
        
        # Color scheme
        colors = {
            'train': 'blue',
            'val': 'red', 
            'test': 'green',  # test set color
            'ignored': 'lightblue',
            'outlier': 'lightgray',
            'robust_outlier': 'pink',
            'fit_line': 'darkgreen'
        }
        
        for idx, benchmark in enumerate(selected_benchmarks, 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # Best model for this benchmark
            best_model_name, best_result = self.get_best_model(benchmark)
            
            if best_result is None:
                ax.text(0.5, 0.5, 'No valid fit', ha='center', va='center')
                ax.set_title(benchmark)
                continue
            
            # Config
            config = best_result['config']
            split_mode = config.get('split_mode', 'percentage')
            use_test_set = config.get('use_test_set', False)
            
            # Data in ExaFLOPs for plotting
            x_peta = best_result['x_all']
            y = best_result['y_all']
            x = self.convert_to_exaflops(x_peta)
            # Max y among all scatter points for this benchmark (matches plotted points)
            scatter_y_max = float(np.nanmax(y))
            
            # Split / outlier masks
            train_mask = best_result['train_mask']
            val_mask = best_result['val_mask']
            test_mask = best_result.get('test_mask', np.zeros(len(x), dtype=bool))
            ignored_mask = best_result['ignored_mask']
            outlier_mask = best_result['outlier_mask']
            robust_mask = best_result.get('robust_mask')
            
            # Scatter points
            x_ignored = x[ignored_mask]
            y_ignored = y[ignored_mask]
            if len(x_ignored) > 0:
                ax.scatter(x_ignored, y_ignored, color=colors['ignored'], s=30, 
                          alpha=0.4, label='Ignored', zorder=2)
            
            x_outliers = x[outlier_mask]
            y_outliers = y[outlier_mask]
            if len(x_outliers) > 0:
                ax.scatter(x_outliers, y_outliers, color=colors['outlier'], s=30, 
                          alpha=0.5, label='Outliers', zorder=3)
            
            if config['use_robust_regression'] and robust_mask is not None:
                train_clean_mask = train_mask & ~outlier_mask
                robust_outlier_mask = train_clean_mask & ~robust_mask
                x_robust_outliers = x[robust_outlier_mask]
                y_robust_outliers = y[robust_outlier_mask]
                if len(x_robust_outliers) > 0:
                    ax.scatter(x_robust_outliers, y_robust_outliers, 
                             color=colors['robust_outlier'], s=30,
                             alpha=0.6, label='LTS Outliers', zorder=4)
                
                x_train = x[robust_mask]
                y_train = y[robust_mask]
            else:
                train_clean_mask = train_mask & ~outlier_mask
                x_train = x[train_clean_mask]
                y_train = y[train_clean_mask]
            
            if len(x_train) > 0:
                ax.scatter(x_train, y_train, color=colors['train'], s=30, 
                          alpha=0.7, label='Train', zorder=5)
            
            x_val = x[val_mask]
            y_val = y[val_mask]
            if len(x_val) > 0:
                ax.scatter(x_val, y_val, color=colors['val'], s=30, 
                          alpha=0.7, label='Val', zorder=5)
            
            # Test points (when enabled)
            if use_test_set:
                x_test = x[test_mask]
                y_test = y[test_mask]
                if len(x_test) > 0:
                    ax.scatter(x_test, y_test, color=colors['test'], s=30,
                              alpha=0.7, label='Test', zorder=5)
            
            # Best-fit curve
            x_dense_peta = np.linspace(x_peta.min(), x_peta.max(), 200)
            y_fit = best_result['func'](x_dense_peta, *best_result['params'])
            x_dense = self.convert_to_exaflops(x_dense_peta)
            ax.plot(
                x_dense,
                y_fit,
                color=colors['fit_line'],
                linewidth=2,
                label=f"{best_model_name}",
                zorder=4,
            )
            
            # Config summary in info box
            config_text = (
                f"Mode: {split_mode}\n"
                f"TR: {self.format_range_for_display(config['train_range'], split_mode)}\n"
                f"VR: {self.format_range_for_display(config['val_range'], split_mode)}\n"
            )
            
            if use_test_set:
                config_text += f"TE: {self.format_range_for_display(config['test_range'], split_mode)}\n"
            
            config_text += f"Robust: {config['use_robust_regression']}\n"
            
            if config.get('fixed_C0') is not None:
                config_text += f"C0: {config['fixed_C0']:.2f} (fixed)\n"
            
            # Info text (Psft: raw performance at RL start, between equation and Train R²)
            psft_line = ""
            if self.psft_by_benchmark:
                psft_val = self.psft_by_benchmark.get(benchmark)
                if psft_val is not None:
                    psft_line = f"Psft: {psft_val:.2f}\n"
            info_text = (f"Model: {best_model_name}\n"
                        f"{best_result['equation']}\n"
                        f"Scatter max: {scatter_y_max:.2f}\n"
                        f"{psft_line}"
                        f"Train R²: {best_result['r2_train']:.3f}\n"
                        f"Val R²: {best_result['r2_val']:.3f}\n"
                        f"Val RMSE: {best_result['rmse_val']:.3f}\n")
            
            if use_test_set and best_result['rmse_test'] is not None:
                info_text += f"Test RMSE: {best_result['rmse_test']:.3f}\n"
            
            info_text += (f"AIC: {best_result['aic']:.1f}\n"
                         f"BIC: {best_result['bic']:.1f}\n"
                         f"---Config---\n{config_text}")
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Axis labels and title
            ax.set_xlabel('Accum RL ExaFLOPs', fontsize=13)
            ax.set_ylabel('Performance', fontsize=13)
            if n_benchmarks > 1:
                ax.set_title(benchmark.replace("_", " ").title(), fontsize=15, fontweight='bold', pad=8)
            ax.tick_params(axis='both', labelsize=11)
            ax.grid(True, alpha=0.3)
            leg_handles, _ = ax.get_legend_handles_labels()
            ax.legend(
                handles=leg_handles,
                fontsize=10,
                loc="lower right",
                ncol=2,
            )
            ax.ticklabel_format(style='plain', axis='x')

            # overall: y-axis from 0 upward so large negative values do not flatten the curve
            if benchmark == 'overall':
                y_stack = np.asarray(np.concatenate([y, np.ravel(y_fit)]), dtype=float)
                y_stack = y_stack[np.isfinite(y_stack)]
                y_hi = float(np.nanmax(y_stack)) if y_stack.size else 1.0
                if not np.isfinite(y_hi):
                    y_hi = 1.0
                y_hi = y_hi * 1.08
                if y_hi <= 0:
                    y_hi = 1.0
                ax.set_ylim(0.0, y_hi)
            
            # Store per-benchmark results
            all_results[benchmark] = {
                'best_model': best_model_name,
                'params': best_result['params'],
                'equation': best_result['equation'],
                'r2_train': best_result['r2_train'],
                'r2_val': best_result['r2_val'],
                'rmse_val': best_result['rmse_val'],
                'r2_test': best_result.get('r2_test'),
                'rmse_test': best_result.get('rmse_test'),
                'aic': best_result['aic'],
                'bic': best_result['bic'],
                'config': config
            }
        
        if n_benchmarks == 1:
            fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.88)
        else:
            plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        
        # Save figure
        if save_path:
            # Ensure output directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Add default extension if missing
            if not os.path.splitext(save_path)[1]:
                save_path = f"{save_path}.{save_format}"
            
            # Write file
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Figure saved to: {save_path}")

        if os.environ.get('FIT_CURVES_NO_SHOW', '').lower() not in ('1', 'true', 'yes'):
            plt.show()
        
        return all_results
    
    def print_summary(self, sft_ckpt, benchmark_filter=None):
        """Print summary table for selected benchmarks."""
        selected_benchmarks = self.select_benchmarks(benchmark_filter)
        
        print(f"\n{'='*130}")
        print(f"SFT Checkpoint {sft_ckpt} - Summary of Best Models")
        print(f"Selected {len(selected_benchmarks)} benchmarks")
        print(f"{'='*130}")
        
        results_table = []
        for benchmark in selected_benchmarks:
            best_model_name, best_result = self.get_best_model(benchmark)
            if best_result:
                config = best_result['config']
                split_mode = config.get('split_mode', 'percentage')
                use_test_set = config.get('use_test_set', False)
                
                row = {
                    'Benchmark': benchmark[:20],  # truncate for column width
                    'Model': best_model_name,
                    'Train R²': f"{best_result['r2_train']:.4f}",
                    'Val R²': f"{best_result['r2_val']:.4f}",
                    'Val RMSE': f"{best_result['rmse_val']:.4f}",
                }
                
                if use_test_set and best_result.get('rmse_test') is not None:
                    row['Test RMSE'] = f"{best_result['rmse_test']:.4f}"
                
                row.update({
                    'AIC': f"{best_result['aic']:.1f}",
                    'BIC': f"{best_result['bic']:.1f}",
                    'Mode': split_mode[:4],  # abbreviated
                    'Config': f"TR:{config['train_range']}"
                })
                
                results_table.append(row)
        
        # Print table
        if results_table:
            col_widths = {}
            for key in results_table[0].keys():
                col_widths[key] = max(len(key), max(len(str(row[key])) for row in results_table))
            
            header = " | ".join(f"{key:{col_widths[key]}}" for key in results_table[0].keys())
            print(header)
            print("-" * len(header))
            
            for row in results_table:
                print(" | ".join(f"{row.get(key, ''):{col_widths[key]}}" for key in results_table[0].keys()))
    
    def predict(self, benchmark_filter=None, predict_flops_list=[40000, 50000, 60000]):
        """Predict performance at future FLOPs for selected benchmarks."""
        selected_benchmarks = self.select_benchmarks(benchmark_filter)
        predictions = {}
        
        for benchmark in selected_benchmarks:
            best_model_name, best_result = self.get_best_model(benchmark)
            if best_result:
                benchmark_predictions = {}
                for flops in predict_flops_list:
                    pred = best_result['func'](flops, *best_result['params'])
                    benchmark_predictions[flops] = pred
                predictions[benchmark] = {
                    'model': best_model_name,
                    'equation': best_result['equation'],
                    'predictions': benchmark_predictions,
                    'config': best_result['config']
                }
        
        return predictions


def plot_advanced_scaling_v2(sft_ckpt, combined_flops2valPerform_zoo_new, branch_flops_zoo,
                            default_config=None, benchmark_configs=None,
                            benchmark_filter=None, predict_flops_list=[40000, 50000, 60000],
                            save_path=None, dpi=300, save_format='png',
                            title_dataset='SFT'):
    """
    Advanced DAPO scaling analysis V2 — per-benchmark config overrides.
    
    Parameters:
    -----------
    sft_ckpt : int
        SFT checkpoint
    combined_flops2valPerform_zoo_new : dict
        Performance data
    branch_flops_zoo : dict
        Branch FLOPs data
    default_config : dict
        Default config for benchmarks without specific overrides
    benchmark_configs : dict
        Per-benchmark configs: {benchmark_name: config_dict, ...}
    benchmark_filter : None, str, list, or callable
        Benchmarks to analyze
    predict_flops_list : list
        FLOPs values to extrapolate to
    
    Example:
    --------
    # Different config per benchmark
    benchmark_configs = {
        'MMLU': {
            'train_range': (0, 50),
            'val_range': (50, 100),
            'use_robust_regression': True,
            'model_types': ['linear', 'log'],
            'metric': 'bic'
        },
        'GPQA': {
            'train_range': (0, 70),
            'val_range': (70, 100),
            'use_robust_regression': False,
            'model_types': 'auto',
            'metric': 'val_rmse'
        }
    }
    """
    
    # Load scatter data
    # print("[DEBUG] check scatters")
    # print(f"sft_ckpt: {sft_ckpt}")
    # print(f"combined_flops2valPerform_zoo_new: {combined_flops2valPerform_zoo_new}")
    # print(f"branch_flops_zoo: {branch_flops_zoo}")
    data = get_rl_scatters(sft_ckpt, combined_flops2valPerform_zoo_new, branch_flops_zoo)
    # print(f"data: {data}")
    
    branch_flops = branch_flops_zoo[sft_ckpt]
    psft_by_benchmark = combined_flops2valPerform_zoo_new[sft_ckpt][branch_flops]

    # print(data)
    # Default config
    if default_config is None:
        default_config = {
            'train_range': (20, 60),
            'val_range': (80, 100),
            'detect_outliers': True,
            'outlier_threshold': 2.5,
            'use_robust_regression': False,
            'lts_alpha': 0.75,
            'model_types': 'auto',
            'metric': 'val_rmse',
            'logistic_optimization': 'adaptive'
        }
    
    # Create fitter
    fitter = AdvancedFLOPsPerformanceFitter_v2(data, default_config, psft_by_benchmark=psft_by_benchmark)
    
    # Apply per-benchmark configs
    if benchmark_configs:
        fitter.set_benchmark_configs_batch(benchmark_configs)
    
    print(f"\nAnalyzing SFT Checkpoint {sft_ckpt}")
    print(f"Default config: {default_config}")
    if benchmark_configs:
        print(f"Benchmark-specific config applied to {len(benchmark_configs)} benchmarks")
    
    # Plot selected benchmarks
    all_results = fitter.plot_selected_benchmarks(sft_ckpt, benchmark_filter,
                                                  save_path=save_path, dpi=dpi, save_format=save_format,
                                                  title_dataset=title_dataset)
    
    # Summary table
    fitter.print_summary(sft_ckpt, benchmark_filter)
    
    # Extrapolation
    predictions = fitter.predict(benchmark_filter, predict_flops_list)
    
    # Print predictions
    if predictions:
        print(f"\n{'='*100}")
        print(f"Predictions for Future FLOPs")
        print(f"{'='*100}")
        for benchmark, pred_info in predictions.items():
            print(f"\n{benchmark} (Model: {pred_info['model']})")
            print(f"Config: TR:{pred_info['config']['train_range']}, "
                  f"RR:{pred_info['config']['use_robust_regression']}")
            print(f"Equation: {pred_info['equation']}")
            for flops, pred_val in pred_info['predictions'].items():
                print(f"  FLOPs={flops}: {pred_val:.4f}")
    
    return all_results, predictions, fitter


# # Usage examples
# if __name__ == "__main__":
#     # Assume combined_flops2valPerform_zoo_new, branch_flops_zoo, get_rl_scatters exist
#     # combined_flops2valPerform_zoo_new, branch_flops_zoo, get_rl_scatters
    
#     sft_ckpts = [0, 360, 720, 1080, 1440, 1800, 3600, 5400, 7200, 9000, 10800, 12600, 14080]
    
#     # Example 1: different configs per benchmark group
#     print(f"\n{'#'*120}")
#     print("Example 1: different configs per benchmark group")
#     print(f"{'#'*120}")
    
#     # Per-benchmark configs
#     benchmark_configs = {
#         # Math benchmarks: robust regression
#         'MATH-500': {
#             'train_range': (0, 60),
#             'val_range': (60, 100),
#             'use_robust_regression': True,
#             'lts_alpha': 0.7,
#             'model_types': ['power', 'logistic'],
#             'metric': 'auto'
#         },
#         'GSM8K': {
#             'train_range': (0, 60),
#             'val_range': (60, 100),
#             'use_robust_regression': True,
#             'lts_alpha': 0.7,
#             'model_types': ['power', 'logistic'],
#             'metric': 'auto'
#         },
        
#         # Language understanding benchmarks: standard fitting
#         'MMLU': {
#             'train_range': (20, 70),
#             'val_range': (70, 100),
#             'use_robust_regression': False,
#             'detect_outliers': True,
#             'outlier_threshold': 3.0,
#             'model_types': 'auto',
#             'metric': 'bic'
#         },
#         'BBH': {
#             'train_range': (20, 70),
#             'val_range': (70, 100),
#             'use_robust_regression': False,
#             'model_types': ['linear', 'log', 'sigmoid'],
#             'metric': 'val_r2'
#         }
#     }
    
#     # Analyze first checkpoint
#     results1, predictions1, fitter1 = plot_advanced_scaling_v2(
#         sft_ckpts[0],
#         combined_flops2valPerform_zoo_new,
#         branch_flops_zoo,
#         default_config={
#             'train_range': (10, 60),
#             'val_range': (60, 100),
#             'detect_outliers': True,
#             'outlier_threshold': 2.5,
#             'use_robust_regression': False,
#             'lts_alpha': 0.75,
#             'model_types': 'auto',
#             'metric': 'val_rmse'
#         },
#         benchmark_configs=benchmark_configs,
#         benchmark_filter=None,  # all benchmarks
#         predict_flops_list=[30000, 40000, 50000, 60000]
#     )
    
#     # Example 2: analyze only specific benchmarks
#     print(f"\n{'#'*120}")
#     print("Example 2: math-related benchmarks only")
#     print(f"{'#'*120}")
    
#     # String filter
#     results2, predictions2, fitter2 = plot_advanced_scaling_v2(
#         sft_ckpts[0],
#         combined_flops2valPerform_zoo_new,
#         branch_flops_zoo,
#         benchmark_filter='MATH',  # benchmarks whose name contains 'MATH'
#         predict_flops_list=[40000, 50000]
#     )
    
#     # Example 3: custom filter function
#     print(f"\n{'#'*120}")
#     print("Example 3: custom filter function")
#     print(f"{'#'*120}")
    
#     # Custom filter: select high-signal benchmarks
#     def custom_filter(benchmark_name):
#         # Extend with domain-specific logic as needed
#         important_benchmarks = ['MMLU', 'GSM8K', 'MATH-500', 'HumanEval', 'GPQA']
#         return benchmark_name in important_benchmarks
    
#     results3, predictions3, fitter3 = plot_advanced_scaling_v2(
#         sft_ckpts[0],
#         combined_flops2valPerform_zoo_new,
#         branch_flops_zoo,
#         benchmark_filter=custom_filter,
#         predict_flops_list=[35000, 45000, 55000]
#     )
    
#     # Example 4: batch config for similar benchmarks
#     print(f"\n{'#'*120}")
#     print("Example 4: batch config for similar benchmarks")
#     print(f"{'#'*120}")
    
#     # List all benchmark names
#     data = get_rl_scatters(sft_ckpts[0], combined_flops2valPerform_zoo_new, branch_flops_zoo)
#     all_benchmarks = list(next(iter(data.values())).keys())
    
#     # Batch configs by benchmark type
#     math_benchmarks = [b for b in all_benchmarks if 'MATH' in b or 'GSM' in b]
#     code_benchmarks = [b for b in all_benchmarks if 'Code' in b or 'HumanEval' in b]
    
#     batch_configs = {}
    
#     # Math benchmark configs
#     for benchmark in math_benchmarks:
#         batch_configs[benchmark] = {
#             'train_range': (0, 65),
#             'val_range': (65, 100),
#             'use_robust_regression': True,
#             'lts_alpha': 0.8,
#             'model_types': ['power', 'logistic', 'exponential'],
#             'metric': 'auto'
#         }
    
#     # Code benchmark configs
#     for benchmark in code_benchmarks:
#         batch_configs[benchmark] = {
#             'train_range': (10, 70),
#             'val_range': (70, 100),
#             'use_robust_regression': False,
#             'model_types': ['linear', 'log'],
#             'metric': 'aic'
#         }
    
#     results4, predictions4, fitter4 = plot_advanced_scaling_v2(
#         sft_ckpts[0],
#         combined_flops2valPerform_zoo_new,
#         branch_flops_zoo,
#         benchmark_configs=batch_configs,
#         benchmark_filter=math_benchmarks + code_benchmarks,  # only configured benchmarks
#         predict_flops_list=[40000, 50000, 60000]
#     )
    
#     # Example 5: interactive exploration — tune config per benchmark
#     print(f"\n{'#'*120}")
#     print("Example 5: interactive exploration for one benchmark")
#     print(f"{'#'*120}")
    
#     # Create fitter instance
#     data = get_rl_scatters(sft_ckpts[0], combined_flops2valPerform_zoo_new, branch_flops_zoo)
#     fitter = AdvancedFLOPsPerformanceFitter(data)
    
#     # Try multiple configs on one benchmark
#     test_benchmark = 'MMLU'
    
#     configs_to_test = [
#         {'name': 'Standard', 'config': {'use_robust_regression': False, 'model_types': 'auto'}},
#         {'name': 'Robust', 'config': {'use_robust_regression': True, 'lts_alpha': 0.75}},
#         {'name': 'Linear Only', 'config': {'model_types': ['linear']}},
#         {'name': 'Nonlinear', 'config': {'model_types': 'nonlinear'}}
#     ]
    
#     print(f"Testing different configs for {test_benchmark}:")
#     for test_config in configs_to_test:
#         fitter.set_benchmark_config(test_benchmark, test_config['config'])
#         best_model, best_result = fitter.get_best_model(test_benchmark)
#         if best_result:
#             print(f"\n{test_config['name']}:")
#             print(f"  Best Model: {best_model}")
#             print(f"  Val R²: {best_result['r2_val']:.4f}")
#             print(f"  Val RMSE: {best_result['rmse_val']:.4f}")
#             print(f"  AIC: {best_result['aic']:.1f}")
