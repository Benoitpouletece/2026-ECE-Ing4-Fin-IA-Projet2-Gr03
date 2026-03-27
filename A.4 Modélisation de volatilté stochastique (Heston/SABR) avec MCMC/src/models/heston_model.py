"""
Modèle probabiliste Heston optimisé pour l'inférence MCMC.

Ce module implémente une version optimisée du modèle Heston avec :
- Vraisemblance basée sur les moments empiriques des rendements
- Priors très informatifs basés sur les connaissances financières
- Reparamétrization pour améliorer la convergence
- Approche simplifiée sans simulation excessive de variables latentes

Le modèle de Heston est défini par :
    dS_t = μ S_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
    dW_t^S · dW_t^v = ρ dt
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Optional, Tuple


class HestonModel:
    """
    Modèle probabiliste Heston optimisé pour l'inférence bayésienne avec NumPyro.
    
    Ce modèle utilise une approche basée sur les moments empiriques et une
    vraisemblance gaussienne conditionnelle pour améliorer la convergence MCMC.
    """
    
    def __init__(
        self,
        S0: float = 100.0,
        dt: float = 1/252,
        mu: Optional[float] = None,
        use_moments_likelihood: bool = True,
        use_conditional_likelihood: bool = False
    ):
        """
        Initialise le modèle Heston optimisé.
        
        Paramètres
        ----------
        S0 : float
            Prix initial de l'actif
        dt : float
            Pas de temps (en années)
        mu : float, optional
            Taux de rendement espéré. Si None, estimé comme paramètre.
        use_moments_likelihood : bool
            Si True, utilise une vraisemblance basée sur les moments empiriques
        use_conditional_likelihood : bool
            Si True, utilise une vraisemblance gaussienne conditionnelle
        """
        self.S0 = S0
        self.dt = dt
        self.mu = mu
        self.use_moments_likelihood = use_moments_likelihood
        self.use_conditional_likelihood = use_conditional_likelihood
    
    def model(self, returns: jnp.ndarray) -> None:
        """
        Définit le modèle probabiliste Heston dans NumPyro.
        
        Ce modèle définit :
        1. Les priors très informatifs pour les paramètres
        2. La vraisemblance basée sur les moments ou conditionnelle
        
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
        # 1. PRIORS TRÈS INFORMATIFS BASÉS SUR LES CONNAISSANCES FINANCIÈRES
        # ============================================================
        
        # Prior pour κ (kappa) - Vitesse de retour à la moyenne
        # Distribution : LogNormal(log(2.0), 0.3) - centré sur 2 avec écart-type réduit
        # Valeur typique : 1.0 à 5.0
        log_kappa = numpyro.sample("log_kappa", dist.Normal(jnp.log(2.0), 0.3))
        kappa = jnp.exp(log_kappa)
        
        # Prior pour θ (theta) - Variance de long terme
        # Distribution : LogNormal(log(0.04), 0.3) - centré sur 0.04 (volatilité ~20%)
        # Valeur typique : 0.01 à 0.10
        log_theta = numpyro.sample("log_theta", dist.Normal(jnp.log(0.04), 0.3))
        theta = jnp.exp(log_theta)
        
        # Prior pour σ (sigma) - Volatilité de la variance
        # Distribution : LogNormal(log(0.3), 0.3) - centré sur 0.3
        # Valeur typique : 0.1 à 0.6
        log_sigma = numpyro.sample("log_sigma", dist.Normal(jnp.log(0.3), 0.3))
        sigma = jnp.exp(log_sigma)
        
        # Prior pour v0 (variance initiale)
        # Distribution : LogNormal(log(0.04), 0.3) - centré sur 0.04
        # Valeur typique : 0.01 à 0.10
        log_v0 = numpyro.sample("log_v0", dist.Normal(jnp.log(0.04), 0.3))
        v0 = jnp.exp(log_v0)
        
        # Prior pour ρ (rho) - Corrélation entre les processus
        # Distribution : TruncatedNormal(-0.7, 0.2, low=-0.99, high=0.99)
        # Plus informatif que Uniform(-1, 1), centré sur une valeur typique négative
        # Valeur typique : -0.8 à -0.2
        rho = numpyro.sample("rho", dist.TruncatedNormal(-0.7, 0.2, low=-0.99, high=0.99))
        
        # Prior pour μ (mu) - Taux de rendement espéré (optionnel)
        if self.mu is None:
            # Distribution : Normal(0.05, 0.1) - centré sur 5% annuel
            # Valeur typique : -0.05 à 0.15
            mu = numpyro.sample("mu", dist.Normal(0.05, 0.1))
        else:
            mu = self.mu
        
        # ============================================================
        # 2. VRAISEMBLANCE BASÉE SUR LES MOMENTS EMPIRIQUES
        # ============================================================
        
        if self.use_moments_likelihood:
            # Calculer les moments empiriques des rendements observés
            # Moyenne des rendements
            empirical_mean = jnp.mean(returns)
            
            # Variance des rendements
            empirical_var = jnp.var(returns)
            
            # Skewness des rendements
            empirical_skew = jnp.mean((returns - empirical_mean)**3) / (jnp.std(returns)**3 + 1e-10)
            
            # Kurtosis des rendements
            empirical_kurt = jnp.mean((returns - empirical_mean)**4) / (jnp.std(returns)**4 + 1e-10)
            
            # Calculer les moments théoriques du modèle Heston
            # Pour un modèle Heston, les moments théoriques sont approximés
            
            # Moment théorique de la moyenne (approximation)
            # E[r_t] ≈ μ*dt - 0.5*E[v_t]*dt
            # Pour un processus CIR en régime stationnaire : E[v_t] = θ
            theoretical_mean = mu * self.dt - 0.5 * theta * self.dt
            
            # Moment théorique de la variance (approximation)
            # Var[r_t] ≈ E[v_t]*dt + O(dt^2)
            # Pour un processus CIR en régime stationnaire : E[v_t] = θ
            theoretical_var = theta * self.dt
            
            # Moment théorique de la skewness (approximation)
            # Skew[r_t] ≈ ρ*σ*sqrt(dt) / sqrt(θ)
            theoretical_skew = rho * sigma * jnp.sqrt(self.dt) / (jnp.sqrt(theta) + 1e-10)
            
            # Moment théorique de la kurtosis (approximation)
            # Kurt[r_t] ≈ 3 + (σ^2*dt) / θ
            theoretical_kurt = 3.0 + (sigma**2 * self.dt) / (theta + 1e-10)
            
            # Vraisemblance basée sur les moments
            # On utilise une distribution normale pour chaque moment
            # avec une variance qui reflète l'incertitude de l'approximation
            
            # Vraisemblance pour la moyenne
            numpyro.sample(
                "obs_mean",
                dist.Normal(theoretical_mean, jnp.sqrt(theoretical_var / n_obs)),
                obs=empirical_mean
            )
            
            # Vraisemblance pour la variance
            numpyro.sample(
                "obs_var",
                dist.Normal(theoretical_var, theoretical_var * 0.5),
                obs=empirical_var
            )
            
            # Vraisemblance pour la skewness
            numpyro.sample(
                "obs_skew",
                dist.Normal(theoretical_skew, 0.5),
                obs=empirical_skew
            )
            
            # Vraisemblance pour la kurtosis
            numpyro.sample(
                "obs_kurt",
                dist.Normal(theoretical_kurt, 1.0),
                obs=empirical_kurt
            )
            
        # ============================================================
        # 3. VRAISEMBLANCE GAUSSIENNE CONDITIONNELLE
        # ============================================================
        
        elif self.use_conditional_likelihood:
            # Approche de quasi-maximum likelihood
            # On simule la variance conditionnelle et on calcule la vraisemblance
            # des rendements conditionnellement à cette variance
            
            # Initialisation de la variance
            v_init = jnp.full(n_paths, v0)
            
            # Simuler la variance avec le processus CIR
            # On utilise une approche déterministe pour éviter la simulation de variables latentes
            def scan_variance(carry, x):
                v_prev = carry
                # Évolution de la variance selon le processus CIR
                # dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
                # On utilise l'espérance conditionnelle : E[v_t | v_{t-1}] = v_{t-1} + κ(θ - v_{t-1}) dt
                dv_mean = kappa * (theta - v_prev) * self.dt
                v_mean = v_prev + dv_mean
                
                # Variance conditionnelle de la variance
                # Var[v_t | v_{t-1}] = σ^2 v_{t-1} dt
                v_var = sigma**2 * jnp.maximum(v_prev, 0.0) * self.dt
                
                # On utilise la valeur moyenne comme approximation
                v_new = jnp.maximum(v_mean, 0.0)
                return v_new, v_new
            
            _, v_sim = jax.lax.scan(scan_variance, v_init, jnp.arange(n_obs))
            v_sim = jnp.concatenate([v_init.reshape(-1, 1), v_sim[:-1].T], axis=1)
            
            # Calculer les rendements attendus et leur volatilité
            expected_returns = mu * self.dt - 0.5 * v_sim * self.dt
            returns_volatility = jnp.sqrt(v_sim * self.dt)
            
            # Vraisemblance des rendements observés
            # Distribution : Normal(expected_returns, returns_volatility)
            numpyro.sample(
                "obs_returns",
                dist.Normal(expected_returns, returns_volatility),
                obs=returns
            )
        
        # ============================================================
        # 4. VRAISEMBLANCE HYBRIDE (COMBINAISON DES DEUX APPROCHES)
        # ============================================================
        
        else:
            # Approche hybride : utiliser les moments pour contraindre les paramètres
            # et une vraisemblance conditionnelle pour les rendements
            
            # Calculer les moments empiriques
            empirical_mean = jnp.mean(returns)
            empirical_var = jnp.var(returns)
            
            # Moments théoriques
            theoretical_mean = mu * self.dt - 0.5 * theta * self.dt
            theoretical_var = theta * self.dt
            
            # Contraintes sur les moments
            numpyro.sample(
                "obs_mean",
                dist.Normal(theoretical_mean, jnp.sqrt(theoretical_var / n_obs)),
                obs=empirical_mean
            )
            
            numpyro.sample(
                "obs_var",
                dist.Normal(theoretical_var, theoretical_var * 0.3),
                obs=empirical_var
            )
            
            # Vraisemblance conditionnelle simplifiée
            # On utilise une variance constante égale à theta
            expected_returns = mu * self.dt - 0.5 * theta * self.dt
            returns_volatility = jnp.sqrt(theta * self.dt)
            
            numpyro.sample(
                "obs_returns",
                dist.Normal(expected_returns, returns_volatility),
                obs=returns
            )
    
    def guide(self, returns: jnp.ndarray) -> None:
        """
        Guide pour l'inférence variationnelle (optionnel).
        
        Ce guide peut être utilisé pour l'inférence variationnelle avec SVI.
        """
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        
        n_paths, n_obs = returns.shape
        
        # Paramètres du guide
        log_kappa_loc = numpyro.param("log_kappa_loc", jnp.log(2.0))
        log_kappa_scale = numpyro.param("log_kappa_scale", 0.3, constraint=dist.constraints.positive)
        
        log_theta_loc = numpyro.param("log_theta_loc", jnp.log(0.04))
        log_theta_scale = numpyro.param("log_theta_scale", 0.3, constraint=dist.constraints.positive)
        
        log_sigma_loc = numpyro.param("log_sigma_loc", jnp.log(0.3))
        log_sigma_scale = numpyro.param("log_sigma_scale", 0.3, constraint=dist.constraints.positive)
        
        log_v0_loc = numpyro.param("log_v0_loc", jnp.log(0.04))
        log_v0_scale = numpyro.param("log_v0_scale", 0.3, constraint=dist.constraints.positive)
        
        rho_loc = numpyro.param("rho_loc", -0.7, constraint=dist.constraints.real)
        rho_scale = numpyro.param("rho_scale", 0.2, constraint=dist.constraints.positive)
        
        # Échantillonnage du guide
        numpyro.sample("log_kappa", dist.Normal(log_kappa_loc, log_kappa_scale))
        numpyro.sample("log_theta", dist.Normal(log_theta_loc, log_theta_scale))
        numpyro.sample("log_sigma", dist.Normal(log_sigma_loc, log_sigma_scale))
        numpyro.sample("log_v0", dist.Normal(log_v0_loc, log_v0_scale))
        numpyro.sample("rho", dist.TruncatedNormal(rho_loc, rho_scale, low=-0.99, high=0.99))
        
        if self.mu is None:
            mu_loc = numpyro.param("mu_loc", 0.05)
            mu_scale = numpyro.param("mu_scale", 0.1, constraint=dist.constraints.positive)
            numpyro.sample("mu", dist.Normal(mu_loc, mu_scale))


class HestonModelConditional:
    """
    Modèle Heston avec vraisemblance gaussienne conditionnelle optimisée.
    
    Ce modèle utilise une approche de quasi-maximum likelihood avec une
    vraisemblance gaussienne conditionnelle pour améliorer la convergence MCMC.
    """
    
    def __init__(
        self,
        S0: float = 100.0,
        dt: float = 1/252,
        mu: Optional[float] = None
    ):
        """
        Initialise le modèle Heston avec vraisemblance conditionnelle.
        
        Paramètres
        ----------
        S0 : float
            Prix initial de l'actif
        dt : float
            Pas de temps (en années)
        mu : float, optional
            Taux de rendement espéré. Si None, estimé comme paramètre.
        """
        self.S0 = S0
        self.dt = dt
        self.mu = mu
    
    def model(self, returns: jnp.ndarray) -> None:
        """
        Définit le modèle probabiliste Heston avec vraisemblance conditionnelle.
        
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
        # PRIORS TRÈS INFORMATIFS
        # ============================================================
        
        # Prior pour κ (kappa) - Vitesse de retour à la moyenne
        log_kappa = numpyro.sample("log_kappa", dist.Normal(jnp.log(2.0), 0.3))
        kappa = jnp.exp(log_kappa)
        
        # Prior pour θ (theta) - Variance de long terme
        log_theta = numpyro.sample("log_theta", dist.Normal(jnp.log(0.04), 0.3))
        theta = jnp.exp(log_theta)
        
        # Prior pour σ (sigma) - Volatilité de la variance
        log_sigma = numpyro.sample("log_sigma", dist.Normal(jnp.log(0.3), 0.3))
        sigma = jnp.exp(log_sigma)
        
        # Prior pour v0 (variance initiale)
        log_v0 = numpyro.sample("log_v0", dist.Normal(jnp.log(0.04), 0.3))
        v0 = jnp.exp(log_v0)
        
        # Prior pour ρ (rho) - Corrélation entre les processus
        rho = numpyro.sample("rho", dist.TruncatedNormal(-0.7, 0.2, low=-0.99, high=0.99))
        
        # Prior pour μ (mu) - Taux de rendement espéré (optionnel)
        if self.mu is None:
            mu = numpyro.sample("mu", dist.Normal(0.05, 0.1))
        else:
            mu = self.mu
        
        # ============================================================
        # VRAISEMBLANCE GAUSSIENNE CONDITIONNELLE SIMPLIFIÉE
        # ============================================================
        
        # Approche simplifiée : on utilise la variance de long terme theta
        # comme approximation de la variance conditionnelle
        
        # Rendements attendus
        expected_returns = mu * self.dt - 0.5 * theta * self.dt
        
        # Volatilité des rendements
        returns_volatility = jnp.sqrt(theta * self.dt)
        
        # Vraisemblance des rendements observés
        numpyro.sample(
            "obs_returns",
            dist.Normal(expected_returns, returns_volatility),
            obs=returns
        )
        
        # Contrainte additionnelle sur la variance empirique
        empirical_var = jnp.var(returns)
        theoretical_var = theta * self.dt
        
        numpyro.sample(
            "obs_var",
            dist.Normal(theoretical_var, theoretical_var * 0.2),
            obs=empirical_var
        )
