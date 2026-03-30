"""
Diagnostics de convergence pour l'inférence MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import jax.numpy as jnp


class MCMCDiagnostics:
    """Classe pour les diagnostics de convergence MCMC."""
    
    def __init__(self, samples: Dict[str, jnp.ndarray]):
        self.samples = samples
        self.diagnostics = {}
    
    def compute_r_hat(self, param_name: str) -> float:
        if param_name not in self.samples:
            raise ValueError(f"Paramètre {param_name} non trouvé dans les échantillons")
        
        param_samples = np.array(self.samples[param_name])
        if param_samples.ndim == 1:
            return np.nan
        
        if param_samples.ndim == 2:
            n_chains, n_samples = param_samples.shape
            chain_means = np.mean(param_samples, axis=1)
            chain_vars = np.var(param_samples, axis=1, ddof=1)
            overall_mean = np.mean(chain_means)
            between_chain_var = n_samples * np.var(chain_means, ddof=1)
            within_chain_var = np.mean(chain_vars)
            var_hat = (n_samples - 1) / n_samples * within_chain_var + \
                      (n_samples + 1) / n_samples * between_chain_var / n_chains
            r_hat = np.sqrt(var_hat / within_chain_var)
            return float(r_hat)
        return np.nan
    
    def compute_ess(self, param_name: str) -> float:
        if param_name not in self.samples:
            raise ValueError(f"Paramètre {param_name} non trouvé dans les échantillons")
        
        param_samples = np.array(self.samples[param_name])
        if param_samples.ndim == 2:
            param_samples = param_samples.flatten()
        
        n_samples = len(param_samples)
        autocorr = self._compute_autocorrelation(param_samples)
        ess = n_samples / (1 + 2 * np.sum(autocorr[1:]))
        return float(ess)
    
    def compute_ess_bulk(self, param_name: str) -> float:
        if param_name not in self.samples:
            raise ValueError(f"Paramètre {param_name} non trouvé dans les échantillons")
        
        param_samples = np.array(self.samples[param_name])
        if param_samples.ndim == 2:
            n_chains, n_samples = param_samples.shape
            ess_per_chain = []
            for i in range(n_chains):
                autocorr = self._compute_autocorrelation(param_samples[i])
                ess = n_samples / (1 + 2 * np.sum(autocorr[1:]))
                ess_per_chain.append(ess)
            return float(np.min(ess_per_chain))
        else:
            n_samples = len(param_samples)
            autocorr = self._compute_autocorrelation(param_samples)
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:]))
            return float(ess)
    
    def _compute_autocorrelation(self, x: np.ndarray, max_lag: int = 100) -> np.ndarray:
        x = x - np.mean(x)
        var = np.var(x)
        if var == 0:
            return np.zeros(max_lag + 1)
        
        autocorr = np.zeros(max_lag + 1)
        autocorr[0] = 1.0
        for lag in range(1, min(max_lag, len(x))):
            autocorr[lag] = np.sum(x[:-lag] * x[lag:]) / (len(x) * var)
        return autocorr
    
    def compute_all_diagnostics(self) -> Dict[str, Dict[str, float]]:
        for param_name in self.samples.keys():
            if param_name not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']:
                self.diagnostics[param_name] = {
                    'r_hat': self.compute_r_hat(param_name),
                    'ess': self.compute_ess_bulk(param_name)
                }
        return self.diagnostics
    
    def print_diagnostics(self):
        if not self.diagnostics:
            self.compute_all_diagnostics()
        
        print("=" * 80)
        print("DIAGNOSTICS DE CONVERGENCE MCMC")
        print("=" * 80)
        
        for param_name, diag in self.diagnostics.items():
            print(f"\n{param_name}:")
            print(f"  R-hat: {diag['r_hat']:.4f}", end="")
            if diag['r_hat'] < 1.1:
                print(" ✓ (Convergence acceptable)")
            elif np.isnan(diag['r_hat']):
                print(" (Une seule chaîne)")
            else:
                print(" ✗ (Convergence insuffisante)")
            
            print(f"  ESS: {diag['ess']:.0f}", end="")
            if diag['ess'] > 1000:
                print(" ✓ (Excellent)")
            elif diag['ess'] > 400:
                print(" (Acceptable)")
            else:
                print(" ✗ (Insuffisant)")
        print("\n" + "=" * 80)
    
    def plot_trace(self, param_names: Optional[List[str]] = None, figsize: tuple = (15, 10)):
        if param_names is None:
            param_names = [k for k in self.samples.keys() if k not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']]
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, param_name in enumerate(param_names):
            if param_name in self.samples:
                param_samples = np.array(self.samples[param_name])
                if param_samples.ndim == 2:
                    n_chains = param_samples.shape[0]
                    for chain in range(n_chains):
                        axes[idx].plot(param_samples[chain, :], alpha=0.6, linewidth=1, label=f'Chaîne {chain+1}')
                    axes[idx].legend()
                else:
                    axes[idx].plot(param_samples, alpha=0.6, linewidth=1)
                axes[idx].set_xlabel('Itération')
                axes[idx].set_ylabel(param_name)
                axes[idx].set_title(f'Trace Plot: {param_name}')
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    def plot_posterior(self, param_names: Optional[List[str]] = None, figsize: tuple = (15, 10)):
        if param_names is None:
            param_names = [k for k in self.samples.keys() if k not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']]
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, param_name in enumerate(param_names):
            if param_name in self.samples:
                param_samples = np.array(self.samples[param_name]).flatten()
                axes[idx].hist(param_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
                axes[idx].axvline(x=np.mean(param_samples), color='r', linestyle='--', label=f'Moyenne = {np.mean(param_samples):.4f}')
                axes[idx].axvline(x=np.median(param_samples), color='g', linestyle='--', label=f'Médiane = {np.median(param_samples):.4f}')
                
                q2_5, q97_5 = np.percentile(param_samples, [2.5, 97.5])
                axes[idx].axvline(x=q2_5, color='b', linestyle=':', alpha=0.5)
                axes[idx].axvline(x=q97_5, color='b', linestyle=':', alpha=0.5, label=f'IC 95%: [{q2_5:.4f}, {q97_5:.4f}]')
                
                axes[idx].set_xlabel(param_name)
                axes[idx].set_ylabel('Densité')
                axes[idx].set_title(f'Distribution Postérieure: {param_name}')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    def plot_autocorrelation(self, param_names: Optional[List[str]] = None, max_lag: int = 50, figsize: tuple = (15, 10)):
        if param_names is None:
            param_names = [k for k in self.samples.keys() if k not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']]
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, param_name in enumerate(param_names):
            if param_name in self.samples:
                param_samples = np.array(self.samples[param_name]).flatten()
                autocorr = self._compute_autocorrelation(param_samples, max_lag)
                axes[idx].stem(range(len(autocorr)), autocorr, basefmt=" ")
                axes[idx].axhline(y=0, color='r', linestyle='-', alpha=0.3)
                axes[idx].set_xlabel('Lag')
                axes[idx].set_ylabel('Autocorrélation')
                axes[idx].set_title(f'Autocorrélation: {param_name}')
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    def check_convergence(self, r_hat_threshold: float = 1.1, ess_threshold: float = 400) -> bool:
        if not self.diagnostics:
            self.compute_all_diagnostics()
        for param_name, diag in self.diagnostics.items():
            if not np.isnan(diag['r_hat']) and diag['r_hat'] > r_hat_threshold:
                return False
            if diag['ess'] < ess_threshold:
                return False
        return True
    
    def get_convergence_report(self) -> str:
        if not self.diagnostics:
            self.compute_all_diagnostics()
        
        report = "=" * 80 + "\nRAPPORT DE CONVERGENCE MCMC\n" + "=" * 80 + "\n\n"
        converged = self.check_convergence()
        
        if converged:
            report += "✓ CONVERGENCE ACCEPTABLE\n\n"
        else:
            report += "✗ CONVERGENCE INSUFFISANTE\n\n"
            
        report += "Détails par paramètre:\n" + "-" * 80 + "\n"
        
        for param_name, diag in self.diagnostics.items():
            report += f"\n{param_name}:\n  R-hat: {diag['r_hat']:.4f}"
            if not np.isnan(diag['r_hat']):
                report += " ✓\n" if diag['r_hat'] < 1.1 else " ✗\n"
            else:
                report += " (une chaîne)\n"
                
            report += f"  ESS: {diag['ess']:.0f}"
            if diag['ess'] > 1000:
                report += " ✓\n"
            elif diag['ess'] > 400:
                report += " (acceptable)\n"
            else:
                report += " ✗\n"
                
        report += "\n" + "=" * 80 + "\n"
        return report