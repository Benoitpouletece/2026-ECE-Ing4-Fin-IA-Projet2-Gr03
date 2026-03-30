"""
Modèle probabiliste Heston optimisé pour l'inférence MCMC.

Ce module implémente une paramétrisation non-centrée (variables latentes)
pour le modèle Heston. Cette approche reconstruit la dynamique temporelle
pas-à-pas, garantissant la bonne convergence de l'algorithme NUTS pour
estimer la corrélation (rho) et le retour à la moyenne (kappa).

Le modèle est défini par :
    dS_t = μ S_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
    dW_t^S · dW_t^v = ρ dt
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Optional


class HestonModel:
    """
    Modèle probabiliste Heston exact pour l'inférence bayésienne avec NumPyro.
    
    Utilise jax.lax.scan pour reconstruire le chemin latent de la variance
    et la décomposition de Cholesky pour isoler l'effet leverage (rho).
    """
    
    def __init__(
        self,
        dt: float = 1/252,
        mu: Optional[float] = None
    ):
        """
        Initialise le modèle Heston.
        
        Paramètres
        ----------
        dt : float
            Pas de temps (en années, ex: 1/252 pour des données journalières)
        mu : float, optional
            Taux de rendement espéré. Si None, estimé comme paramètre.
        """
        self.dt = dt
        self.mu = mu
    
    def model(self, returns: jnp.ndarray) -> None:
        """
        Définit le modèle probabiliste Heston avec variables latentes.
        
        Paramètres
        ----------
        returns : jnp.ndarray
            Rendements observés (logistiques)
            Shape : (n_observations,) ou (n_paths, n_observations)
        """
        # Vérification de la forme des données
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        
        n_paths, n_obs = returns.shape
        
        # ============================================================
        # 1. PRIORS STRICTEMENT POSITIFS ET INFORMATIFS
        # ============================================================
        
        # kappa : Vitesse de retour à la moyenne (doit être > 0)
        kappa = numpyro.sample("kappa", dist.TruncatedNormal(2.0, 1.0, low=0.01))
        
        # theta : Variance de long terme (doit être > 0)
        theta = numpyro.sample("theta", dist.TruncatedNormal(0.04, 0.02, low=0.001))
        
        # sigma : Volatilité de la variance (doit être > 0)
        sigma = numpyro.sample("sigma", dist.TruncatedNormal(0.3, 0.2, low=0.01))
        
        # v0 : Variance initiale
        v0 = numpyro.sample("v0", dist.TruncatedNormal(0.04, 0.02, low=0.001))
        
        # rho : Corrélation (bornée entre -1 et 1, priorisée autour de valeurs typiques)
        rho = numpyro.sample("rho", dist.TruncatedNormal(-0.7, 0.2, low=-0.99, high=0.99))
        
        # mu : Taux de rendement espéré (optionnel)
        if self.mu is None:
            mu = numpyro.sample("mu", dist.Normal(0.05, 0.1))
        else:
            mu = self.mu

        # ============================================================
        # 2. VARIABLES LATENTES (CHOCS STOCHASTIQUES)
        # ============================================================
        
        # On échantillonne les chocs gaussiens Z_v qui dirigent la variance.
        # Shape: (n_obs, n_paths) pour itérer correctement sur l'axe du temps avec lax.scan
        with numpyro.plate("paths", n_paths):
            with numpyro.plate("time", n_obs):
                Z_v = numpyro.sample("Z_v", dist.Normal(0.0, 1.0))
        
        # ============================================================
        # 3. RECONSTRUCTION SÉQUENTIELLE DE LA VARIANCE (EULER-MARUYAMA)
        # ============================================================
        
        def transition_fn(v_prev, z_v):
            # Sécurité absolue : la variance ne peut pas être négative ou nulle
            v_safe = jnp.maximum(v_prev, 1e-8)
            
            # Dérive (Drift) : kappa * (theta - v_t) * dt
            drift = kappa * (theta - v_safe) * self.dt
            
            # Diffusion : sigma * sqrt(v_t) * dW_v (où dW_v = z_v * sqrt(dt))
            diffusion = sigma * jnp.sqrt(v_safe) * jnp.sqrt(self.dt) * z_v
            
            # Évolution
            v_next = v_safe + drift + diffusion
            
            # Réflexion à zéro pour éviter les instabilités numériques
            v_next = jnp.maximum(v_next, 1e-8)
            
            return v_next, v_safe
        
        # Initialisation de la variance pour chaque trajectoire
        v_init = jnp.full(n_paths, v0)
        
        # Exécution rapide et compilée sur l'axe du temps (scan)
        _, v_path_T = jax.lax.scan(transition_fn, v_init, Z_v)
        
        # Transposition pour aligner avec les rendements : shape (n_paths, n_obs)
        v_path = v_path_T.T
        Z_v_paths = Z_v.T
        
        # ============================================================
        # 4. VRAISEMBLANCE CONDITIONNELLE DES RENDEMENTS (PRIX)
        # ============================================================
        
        # Grâce à la décomposition de Cholesky, le mouvement brownien du prix 
        # s'écrit : dW_S = rho * dW_v + sqrt(1 - rho^2) * dW_perp
        # Le rendement attendu conditionnellement au choc de la variance (Z_v) est donc :
        expected_returns = (mu - 0.5 * v_path) * self.dt + rho * jnp.sqrt(v_path * self.dt) * Z_v_paths
        
        # La volatilité résiduelle du rendement dépend de la part non corrélée : sqrt(1 - rho^2)
        vol_returns = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-6) * v_path * self.dt)
        
        # Observation des rendements
        numpyro.sample(
            "obs_returns",
            dist.Normal(expected_returns, vol_returns),
            obs=returns
        )