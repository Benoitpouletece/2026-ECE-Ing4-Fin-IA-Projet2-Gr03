"""
Simulation du modèle SABR (Stochastic Alpha Beta Rho).

Le modèle SABR est défini par :
    dF_t = α_t F_t^β dW_t^F
    dα_t = ν α_t dW_t^α
    dW_t^F · dW_t^α = ρ dt
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class SABRSimulator:
    """
    Simulateur pour le modèle SABR.
    Utilise une discrétisation d'Euler-Maruyama pour le Forward, 
    et la solution exacte (log-normale) pour la volatilité.
    """
    
    def __init__(
        self,
        F0: float = 0.05,    # Taux forward initial (ex: 5%)
        alpha0: float = 0.2, # Volatilité initiale
        beta: float = 0.5,   # Élasticité (0.5 = type CIR)
        nu: float = 0.4,     # Volatilité de la volatilité
        rho: float = -0.5,   # Corrélation
        T: float = 1.0,
        dt: float = 1/252,
        seed: Optional[int] = None
    ):
        self.F0 = F0
        self.alpha0 = alpha0
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.seed = seed
        
        self.F = None
        self.alpha = None
        self.t = None
    
    def simulate(self, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.t = np.linspace(0, self.T, self.n_steps + 1)
        self.F = np.zeros((self.n_steps + 1, n_paths))
        self.alpha = np.zeros((self.n_steps + 1, n_paths))
        
        self.F[0, :] = self.F0
        self.alpha[0, :] = self.alpha0
        
        # Chocs corrélés
        Z1 = np.random.randn(self.n_steps, n_paths)
        Z2 = np.random.randn(self.n_steps, n_paths)
        
        dW_F = np.sqrt(self.dt) * Z1
        dW_alpha = np.sqrt(self.dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        for i in range(1, self.n_steps + 1):
            F_prev = self.F[i-1, :]
            alpha_prev = self.alpha[i-1, :]
            
            # Dynamique exacte de la volatilité (log-normale, stricte positivité garantie)
            self.alpha[i, :] = alpha_prev * np.exp(-0.5 * self.nu**2 * self.dt + self.nu * dW_alpha[i-1, :])
            
            # Dynamique du Forward (Euler-Maruyama avec protection contre les taux négatifs)
            F_safe = np.maximum(F_prev, 1e-8)
            dF = alpha_prev * (F_safe**self.beta) * dW_F[i-1, :]
            
            self.F[i, :] = np.maximum(F_safe + dF, 1e-8) # Absorbtion à zéro
            
        return self.F, self.alpha, self.t

    def get_forward_differences(self) -> np.ndarray:
        """Retourne dF_t = F_t - F_{t-1}"""
        if self.F is None:
            raise ValueError("Appelez simulate() d'abord.")
        return np.diff(self.F, axis=0)