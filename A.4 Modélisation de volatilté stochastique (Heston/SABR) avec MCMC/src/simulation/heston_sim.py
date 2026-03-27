"""
Simulation du modèle de Heston utilisant la méthode d'Euler-Maruyama.

Le modèle de Heston est défini par :
    dS_t = μ S_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
    dW_t^S · dW_t^v = ρ dt
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import norm


class HestonSimulator:
    """
    Simulateur pour le modèle de volatilité stochastique de Heston.
    
    Utilise la méthode d'Euler-Maruyama pour discrétiser les EDS.
    """
    
    def __init__(
        self,
        S0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        T: float = 1.0,
        dt: float = 1/252,
        seed: Optional[int] = None
    ):
        """
        Initialise le simulateur Heston.
        
        Paramètres
        ----------
        S0 : float
            Prix initial de l'actif
        v0 : float
            Variance initiale
        mu : float
            Taux de rendement espéré (drift)
        kappa : float
            Vitesse de retour à la moyenne
        theta : float
            Variance de long terme
        sigma : float
            Volatilité de la variance (vol of vol)
        rho : float
            Corrélation entre les processus
        T : float
            Horizon temporel (en années)
        dt : float
            Pas de temps (en années)
        seed : int, optional
            Graine pour la reproductibilité
        """
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.seed = seed
        
        # Vérification de la condition de Feller
        self.feller_condition = 2 * kappa * theta >= sigma**2
        if not self.feller_condition:
            print(f"Attention: Condition de Feller non satisfaite (2κθ = {2*kappa*theta:.4f} < σ² = {sigma**2:.4f})")
            print("La variance peut atteindre zéro.")
        
        # Stockage des résultats
        self.S = None
        self.v = None
        self.t = None
    
    def simulate(self, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simule des trajectoires du modèle de Heston.
        
        Paramètres
        ----------
        n_paths : int
            Nombre de trajectoires à simuler
            
        Retourne
        -------
        S : np.ndarray, shape (n_steps+1, n_paths)
            Trajectoires des prix
        v : np.ndarray, shape (n_steps+1, n_paths)
            Trajectoires de la variance
        t : np.ndarray, shape (n_steps+1,)
            Grille temporelle
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Initialisation
        self.t = np.linspace(0, self.T, self.n_steps + 1)
        self.S = np.zeros((self.n_steps + 1, n_paths))
        self.v = np.zeros((self.n_steps + 1, n_paths))
        
        self.S[0, :] = self.S0
        self.v[0, :] = self.v0
        
        # Génération des variables aléatoires corrélées
        # dW^S et dW^v sont corrélés avec coefficient rho
        # On utilise la décomposition de Cholesky
        Z1 = np.random.randn(self.n_steps, n_paths)
        Z2 = np.random.randn(self.n_steps, n_paths)
        
        dW_S = np.sqrt(self.dt) * Z1
        dW_v = np.sqrt(self.dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        # Schéma d'Euler-Maruyama
        for i in range(1, self.n_steps + 1):
            # Équation de la variance (processus CIR)
            # dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
            v_prev = self.v[i-1, :]
            
            # Assurer la positivité de la variance
            sqrt_v = np.sqrt(np.maximum(v_prev, 0))
            
            dv = self.kappa * (self.theta - v_prev) * self.dt + \
                 self.sigma * sqrt_v * dW_v[i-1, :]
            
            self.v[i, :] = np.maximum(v_prev + dv, 0)  # Reflection à zéro
            
            # Équation du prix
            # dS_t = μ S_t dt + √v_t S_t dW_t^S
            S_prev = self.S[i-1, :]
            sqrt_v_current = np.sqrt(self.v[i-1, :])
            
            dS = self.mu * S_prev * self.dt + \
                 sqrt_v_current * S_prev * dW_S[i-1, :]
            
            self.S[i, :] = S_prev + dS
        
        return self.S, self.v, self.t
    
    def get_returns(self) -> np.ndarray:
        """
        Calcule les rendements logistiques des prix simulés.
        
        Retourne
        -------
        returns : np.ndarray
            Rendements logistiques
        """
        if self.S is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        return np.diff(np.log(self.S), axis=0)
    
    def get_volatility(self) -> np.ndarray:
        """
        Retourne la volatilité (racine carrée de la variance).
        
        Retourne
        -------
        volatility : np.ndarray
            Volatilité instantanée
        """
        if self.v is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        return np.sqrt(self.v)
    
    def plot_paths(
        self,
        n_paths_to_plot: int = 10,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualise les trajectoires simulées.
        
        Paramètres
        ----------
        n_paths_to_plot : int
            Nombre de trajectoires à afficher
        figsize : tuple
            Taille de la figure
        save_path : str, optional
            Chemin pour sauvegarder la figure
        """
        if self.S is None or self.v is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        n_paths = min(n_paths_to_plot, self.S.shape[1])
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. Trajectoires des prix
        axes[0].plot(self.t, self.S[:, :n_paths], alpha=0.6, linewidth=1)
        axes[0].set_xlabel('Temps (années)')
        axes[0].set_ylabel('Prix')
        axes[0].set_title(f'Trajectoires des Prix - Modèle de Heston (n={n_paths})')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Trajectoires de la variance
        axes[1].plot(self.t, self.v[:, :n_paths], alpha=0.6, linewidth=1)
        axes[1].axhline(y=self.theta, color='r', linestyle='--', 
                       label=f'Variance long terme (θ={self.theta:.4f})')
        axes[1].set_xlabel('Temps (années)')
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Trajectoires de la Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Trajectoires de la volatilité
        volatility = self.get_volatility()
        axes[2].plot(self.t, volatility[:, :n_paths], alpha=0.6, linewidth=1)
        axes[2].axhline(y=np.sqrt(self.theta), color='r', linestyle='--',
                       label=f'Volatilité long terme (√θ={np.sqrt(self.theta):.4f})')
        axes[2].set_xlabel('Temps (années)')
        axes[2].set_ylabel('Volatilité')
        axes[2].set_title('Trajectoires de la Volatilité')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_single_path(
        self,
        path_idx: int = 0,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None
    ):
        """
        Visualise une trajectoire unique avec plus de détails.
        
        Paramètres
        ----------
        path_idx : int
            Index de la trajectoire à afficher
        figsize : tuple
            Taille de la figure
        save_path : str, optional
            Chemin pour sauvegarder la figure
        """
        if self.S is None or self.v is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        if path_idx >= self.S.shape[1]:
            raise ValueError(f"Index {path_idx} hors limites. Nombre de trajectoires: {self.S.shape[1]}")
        
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        
        # 1. Prix
        axes[0, 0].plot(self.t, self.S[:, path_idx], linewidth=2)
        axes[0, 0].set_xlabel('Temps (années)')
        axes[0, 0].set_ylabel('Prix')
        axes[0, 0].set_title('Trajectoire du Prix')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Variance
        axes[0, 1].plot(self.t, self.v[:, path_idx], linewidth=2, color='orange')
        axes[0, 1].axhline(y=self.theta, color='r', linestyle='--', 
                          label=f'θ = {self.theta:.4f}')
        axes[0, 1].set_xlabel('Temps (années)')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title('Trajectoire de la Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Volatilité
        volatility = self.get_volatility()
        axes[1, 0].plot(self.t, volatility[:, path_idx], linewidth=2, color='green')
        axes[1, 0].axhline(y=np.sqrt(self.theta), color='r', linestyle='--',
                          label=f'√θ = {np.sqrt(self.theta):.4f}')
        axes[1, 0].set_xlabel('Temps (années)')
        axes[1, 0].set_ylabel('Volatilité')
        axes[1, 0].set_title('Trajectoire de la Volatilité')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rendements
        returns = self.get_returns()
        axes[1, 1].plot(self.t[1:], returns[:, path_idx], linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Temps (années)')
        axes[1, 1].set_ylabel('Rendement')
        axes[1, 1].set_title('Rendements Logistiques')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Histogramme des rendements
        axes[2, 0].hist(returns[:, path_idx], bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[2, 0].axvline(x=np.mean(returns[:, path_idx]), color='r', linestyle='--',
                          label=f'Moyenne = {np.mean(returns[:, path_idx]):.4f}')
        axes[2, 0].axvline(x=np.std(returns[:, path_idx]), color='g', linestyle='--',
                          label=f'Écart-type = {np.std(returns[:, path_idx]):.4f}')
        axes[2, 0].axvline(x=-np.std(returns[:, path_idx]), color='g', linestyle='--')
        axes[2, 0].set_xlabel('Rendement')
        axes[2, 0].set_ylabel('Densité')
        axes[2, 0].set_title('Distribution des Rendements')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Histogramme de la variance
        axes[2, 1].hist(self.v[:, path_idx], bins=50, density=True, alpha=0.7, 
                       edgecolor='black', color='orange')
        axes[2, 1].axvline(x=np.mean(self.v[:, path_idx]), color='r', linestyle='--',
                          label=f'Moyenne = {np.mean(self.v[:, path_idx]):.4f}')
        axes[2, 1].axvline(x=self.theta, color='g', linestyle='--',
                          label=f'θ = {self.theta:.4f}')
        axes[2, 1].set_xlabel('Variance')
        axes[2, 1].set_ylabel('Densité')
        axes[2, 1].set_title('Distribution de la Variance')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. QQ-plot des rendements
        from scipy import stats
        stats.probplot(returns[:, path_idx], dist="norm", plot=axes[3, 0])
        axes[3, 0].set_title('QQ-plot des Rendements (vs Normale)')
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. Volatilité vs Rendements (effet leverage)
        axes[3, 1].scatter(returns[:, path_idx], volatility[1:, path_idx], 
                          alpha=0.5, s=10)
        axes[3, 1].set_xlabel('Rendement')
        axes[3, 1].set_ylabel('Volatilité')
        axes[3, 1].set_title(f'Effet Leverage (ρ = {self.rho:.2f})')
        axes[3, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_implied_volatility_surface(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ):
        """
        Visualise la surface de volatilité implicite approximative.
        
        Note: Ceci est une approximation basée sur la volatilité locale.
        Pour une vraie surface de volatilité implicite, il faut utiliser
        la formule de Heston pour le pricing d'options.
        
        Paramètres
        ----------
        strikes : np.ndarray, optional
            Strikes à considérer
        maturities : np.ndarray, optional
            Maturités à considérer
        figsize : tuple
            Taille de la figure
        save_path : str, optional
            Chemin pour sauvegarder la figure
        """
        if self.v is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        # Valeurs par défaut
        if strikes is None:
            S_mean = np.mean(self.S)
            strikes = np.linspace(0.7 * S_mean, 1.3 * S_mean, 20)
        
        if maturities is None:
            maturities = np.linspace(0.1, self.T, 10)
        
        # Créer une grille
        K, T = np.meshgrid(strikes, maturities)
        
        # Approximation de la volatilité implicite
        # Ceci est une simplification - la vraie formule de Heston est plus complexe
        vol_surface = np.zeros_like(K)
        
        for i, t in enumerate(maturities):
            idx = min(int(t / self.dt), self.n_steps)
            v_avg = np.mean(self.v[:idx, 0])  # Variance moyenne jusqu'à t
            
            # Approximation : volatilité implicite ≈ √(variance moyenne)
            # avec ajustement pour le moneyness
            for j, k in enumerate(strikes):
                moneyness = k / self.S0
                # Ajustement simple pour le skew
                skew_adjustment = 1 + self.rho * np.log(moneyness)
                vol_surface[i, j] = np.sqrt(v_avg) * skew_adjustment
        
        # Visualisation
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(K, T, vol_surface, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturité')
        ax.set_zlabel('Volatilité Implicite')
        ax.set_title('Surface de Volatilité Implicite (Approximation)')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")
        
        plt.show()
    
    def summary(self) -> dict:
        """
        Retourne un résumé statistique des simulations.
        
        Retourne
        -------
        summary : dict
            Dictionnaire contenant les statistiques
        """
        if self.S is None or self.v is None:
            raise ValueError("Aucune simulation n'a été effectuée. Appelez simulate() d'abord.")
        
        returns = self.get_returns()
        volatility = self.get_volatility()
        
        summary = {
            'paramètres': {
                'S0': self.S0,
                'v0': self.v0,
                'mu': self.mu,
                'kappa': self.kappa,
                'theta': self.theta,
                'sigma': self.sigma,
                'rho': self.rho,
                'T': self.T,
                'dt': self.dt,
                'n_steps': self.n_steps,
                'n_paths': self.S.shape[1],
                'feller_condition': self.feller_condition
            },
            'prix': {
                'moyenne': np.mean(self.S),
                'écart-type': np.std(self.S),
                'min': np.min(self.S),
                'max': np.max(self.S),
                'final': self.S[-1, :].mean()
            },
            'variance': {
                'moyenne': np.mean(self.v),
                'écart-type': np.std(self.v),
                'min': np.min(self.v),
                'max': np.max(self.v),
                'final': self.v[-1, :].mean()
            },
            'volatilité': {
                'moyenne': np.mean(volatility),
                'écart-type': np.std(volatility),
                'min': np.min(volatility),
                'max': np.max(volatility)
            },
            'rendements': {
                'moyenne': np.mean(returns),
                'écart-type': np.std(returns),
                'skewness': float(stats.skew(returns.flatten())) if hasattr(stats, 'skew') else None,
                'kurtosis': float(stats.kurtosis(returns.flatten())) if hasattr(stats, 'kurtosis') else None
            }
        }
        
        return summary
    
    def print_summary(self):
        """Affiche le résumé statistique."""
        summary = self.summary()
        
        print("=" * 60)
        print("RÉSUMÉ DE LA SIMULATION HESTON")
        print("=" * 60)
        
        print("\nParamètres:")
        for key, value in summary['paramètres'].items():
            print(f"  {key}: {value}")
        
        print("\nStatistiques des Prix:")
        for key, value in summary['prix'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nStatistiques de la Variance:")
        for key, value in summary['variance'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nStatistiques de la Volatilité:")
        for key, value in summary['volatilité'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nStatistiques des Rendements:")
        for key, value in summary['rendements'].items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
        
        print("=" * 60)


# Import pour les statistiques
from scipy import stats