"""
Échantillonneur MCMC pour l'inférence bayésienne avec NumPyro.

Ce module implémente l'inférence MCMC en utilisant l'algorithme NUTS (No-U-Turn Sampler)
de NumPyro pour estimer les paramètres des modèles de volatilité stochastique.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from typing import Optional, Dict, Any, Tuple
import time


class MCMCSampler:
    """
    Échantillonneur MCMC utilisant l'algorithme NUTS de NumPyro.
    """
    
    def __init__(
        self,
        model,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        chain_method: str = 'parallel',
        progress_bar: bool = True,
        jit_model_args: bool = False,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        init_strategy: str = 'uniform'
    ):
        self.model = model
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.chain_method = chain_method
        self.progress_bar = progress_bar
        self.jit_model_args = jit_model_args
        self.target_accept_prob = target_accept_prob
        self.max_tree_depth = max_tree_depth
        self.init_strategy = init_strategy
        
        self.mcmc = None
        self.samples = None
        self.inference_time = None
        
    def _get_init_strategy(self):
        if self.init_strategy == 'uniform':
            return numpyro.infer.init_to_uniform()
        elif self.init_strategy == 'prior':
            return numpyro.infer.init_to_sample()
        elif self.init_strategy == 'adapt_diag':
            return numpyro.infer.init_to_median()
        else:
            return numpyro.infer.init_to_uniform()
    
    def run(self, rng_key: jax.random.PRNGKey, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        print(f"Configuration MCMC:")
        print(f"  Warm-up: {self.num_warmup}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Chains: {self.num_chains}")
        print(f"  Chain method: {self.chain_method}")
        print(f"  Target accept prob: {self.target_accept_prob}")
        print(f"  Max tree depth: {self.max_tree_depth}")
        print(f"  Init strategy: {self.init_strategy}\n")
        
        nuts_kernel = NUTS(
            self.model,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            init_strategy=self._get_init_strategy()
        )
        
        self.mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method=self.chain_method,
            progress_bar=self.progress_bar,
            jit_model_args=self.jit_model_args
        )
        
        print("Démarrage de l'inférence MCMC...")
        start_time = time.time()
        self.mcmc.run(rng_key, *args, **kwargs)
        self.inference_time = time.time() - start_time
        print(f"Inférence terminée en {self.inference_time:.2f} secondes")
        
        self.samples = self.mcmc.get_samples(group_by_chain=True)
        
        print("\n=== DEBUG: Structure des échantillons ===")
        for param_name, param_samples in self.samples.items():
            if param_name not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']:
                print(f"{param_name}: shape = {param_samples.shape}, ndim = {param_samples.ndim}")
        print("=== FIN DEBUG ===\n")
        
        return self.samples
    
    def get_samples(self) -> Dict[str, jnp.ndarray]:
        if self.samples is None:
            raise ValueError("Aucune inférence n'a été effectuée. Appelez run() d'abord.")
        return self.samples
    
    def get_posterior_summary(self) -> Dict[str, Dict[str, float]]:
        if self.mcmc is None:
            raise ValueError("Aucune inférence n'a été effectuée. Appelez run() d'abord.")
        
        summary = {}
        samples = self.get_samples()
        
        for param_name, param_samples in samples.items():
            if param_name not in ['Z_v', 'Z_alpha', 'obs_returns', 'obs_dF', 'obs']:
                param_samples_np = np.array(param_samples)
                
                if param_samples_np.ndim == 2:
                    param_samples_flat = param_samples_np.flatten()
                else:
                    param_samples_flat = param_samples_np
                
                summary[param_name] = {
                    'mean': float(np.mean(param_samples_flat)),
                    'std': float(np.std(param_samples_flat)),
                    'median': float(np.median(param_samples_flat)),
                    'q2.5': float(np.percentile(param_samples_flat, 2.5)),
                    'q25': float(np.percentile(param_samples_flat, 25)),
                    'q75': float(np.percentile(param_samples_flat, 75)),
                    'q97.5': float(np.percentile(param_samples_flat, 97.5)),
                    'n_eff': None,
                    'r_hat': None
                }
        return summary
    
    def print_summary(self):
        if self.mcmc is None:
            raise ValueError("Aucune inférence n'a été effectuée. Appelez run() d'abord.")
        
        print("=" * 80)
        print("RÉSUMÉ DE L'INFÉRENCE MCMC")
        print("=" * 80)
        
        print(f"\nConfiguration:")
        print(f"  Warm-up: {self.num_warmup}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Chains: {self.num_chains}")
        print(f"  Temps d'inférence: {self.inference_time:.2f} secondes")
        
        print(f"\nStatistiques Postérieures:")
        print("-" * 80)
        
        summary = self.get_posterior_summary()
        for param_name, stats in summary.items():
            print(f"\n{param_name}:")
            print(f"  Moyenne: {stats['mean']:.6f}")
            print(f"  Écart-type: {stats['std']:.6f}")
            print(f"  Médiane: {stats['median']:.6f}")
            print(f"  IC 95%: [{stats['q2.5']:.6f}, {stats['q97.5']:.6f}]")
        print("\n" + "=" * 80)
    
    def compare_with_true_params(self, true_params: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        if self.mcmc is None:
            raise ValueError("Aucune inférence n'a été effectuée. Appelez run() d'abord.")
        
        summary = self.get_posterior_summary()
        comparison = {}
        
        for param_name, true_value in true_params.items():
            if param_name in summary:
                stats = summary[param_name]
                comparison[param_name] = {
                    'true': true_value,
                    'estimated': stats['mean'],
                    'error': stats['mean'] - true_value,
                    'relative_error': abs(stats['mean'] - true_value) / (abs(true_value) + 1e-10) * 100,
                    'in_ci': stats['q2.5'] <= true_value <= stats['q97.5']
                }
        return comparison
    
    def print_comparison(self, true_params: Dict[str, float]):
        comparison = self.compare_with_true_params(true_params)
        print("=" * 80)
        print("COMPARAISON AVEC LES VRAIS PARAMÈTRES")
        print("=" * 80)
        for param_name, comp in comparison.items():
            print(f"\n{param_name}:")
            print(f"  Vrai: {comp['true']:.6f}")
            print(f"  Estimé: {comp['estimated']:.6f}")
            print(f"  Erreur: {comp['error']:.6f}")
            print(f"  Erreur relative: {comp['relative_error']:.2f}%")
            print(f"  Dans IC 95%: {'✓' if comp['in_ci'] else '✗'}")
        print("\n" + "=" * 80)
    
    def get_inference_data(self):
        if self.mcmc is None:
            raise ValueError("Aucune inférence n'a été effectuée. Appelez run() d'abord.")
        try:
            import arviz as az
            return az.from_numpyro(self.mcmc)
        except ImportError:
            print("ArviZ n'est pas installé. Installez-le avec: pip install arviz")
            return None


class MCMCSamplerConfig:
    """Configuration prédéfinie pour l'échantillonneur MCMC."""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            'num_warmup': 1000,
            'num_samples': 2000,
            'num_chains': 4,
            'chain_method': 'parallel',
            'progress_bar': True,
            'jit_model_args': False,
            'target_accept_prob': 0.95,
            'max_tree_depth': 10,
            'init_strategy': 'uniform'
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        return {
            'num_warmup': 500,
            'num_samples': 500,
            'num_chains': 2,
            'chain_method': 'parallel',
            'progress_bar': True,
            'jit_model_args': False,
            'target_accept_prob': 0.95,
            'max_tree_depth': 8,
            'init_strategy': 'uniform'
        }
    
    @staticmethod
    def get_high_quality_config() -> Dict[str, Any]:
        return {
            'num_warmup': 2000,
            'num_samples': 4000,
            'num_chains': 4,
            'chain_method': 'parallel',
            'progress_bar': True,
            'jit_model_args': True,
            'target_accept_prob': 0.95,
            'max_tree_depth': 12,
            'init_strategy': 'adapt_diag'
        }