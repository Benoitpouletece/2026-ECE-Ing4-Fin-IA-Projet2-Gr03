"""
Modèle probabiliste SABR implémenté avec NumPyro.
Utilise la paramétrisation non-centrée avec variables latentes pour l'inférence MCMC.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Optional

class SABRModel:
    """
    Modèle probabiliste SABR exact pour l'inférence bayésienne avec NumPyro.
    """
    def __init__(self, dt: float = 1/252):
        self.dt = dt
        
    def model(self, F_obs: jnp.ndarray) -> None:
        """
        Paramètres
        ----------
        F_obs : jnp.ndarray
            Taux forward observés. Shape : (n_paths, n_obs)
        """
        if F_obs.ndim == 1:
            F_obs = F_obs.reshape(1, -1)
            
        n_paths, n_obs = F_obs.shape
        n_transitions = n_obs - 1
        
        # 1. Priors SABR
        alpha0 = numpyro.sample("alpha0", dist.TruncatedNormal(0.2, 0.1, low=0.01))
        beta = numpyro.sample("beta", dist.Uniform(0.0, 1.0))
        nu = numpyro.sample("nu", dist.TruncatedNormal(0.4, 0.2, low=0.01))
        rho = numpyro.sample("rho", dist.TruncatedNormal(-0.5, 0.2, low=-0.99, high=0.99))
        
        # 2. Variables latentes pour la volatilité (Z_alpha)
        with numpyro.plate("paths", n_paths):
            with numpyro.plate("time", n_transitions):
                Z_alpha = numpyro.sample("Z_alpha", dist.Normal(0.0, 1.0))
                
        # 3. Reconstruction de la trajectoire de volatilité (Exact GBM)
        def transition_fn(alpha_prev, z_alpha):
            # Formule exacte pour éviter les volatilités négatives
            alpha_next = alpha_prev * jnp.exp(-0.5 * nu**2 * self.dt + nu * jnp.sqrt(self.dt) * z_alpha)
            return alpha_next, alpha_prev
            
        alpha_init = jnp.full(n_paths, alpha0)
        _, alpha_path_T = jax.lax.scan(transition_fn, alpha_init, Z_alpha)
        alpha_path = alpha_path_T.T
        Z_alpha_paths = Z_alpha.T
        
        # 4. Vraisemblance conditionnelle sur les différences du Forward (dF)
        dF = F_obs[:, 1:] - F_obs[:, :-1]
        F_prev = F_obs[:, :-1]
        F_prev_safe = jnp.maximum(F_prev, 1e-8)
        
        # Effet leverage via Cholesky : dW_F = rho * dW_alpha + sqrt(1-rho^2) * dW_perp
        expected_dF = rho * alpha_path * (F_prev_safe**beta) * jnp.sqrt(self.dt) * Z_alpha_paths
        vol_dF = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-6)) * alpha_path * (F_prev_safe**beta) * jnp.sqrt(self.dt)
        
        numpyro.sample(
            "obs_dF",
            dist.Normal(expected_dF, vol_dF),
            obs=dF
        )