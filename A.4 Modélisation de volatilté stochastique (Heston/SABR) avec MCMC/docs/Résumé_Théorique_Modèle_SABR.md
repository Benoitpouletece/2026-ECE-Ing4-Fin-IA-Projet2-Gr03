# Résumé Théorique du Modèle SABR 

Le modèle SABR (Stochastic Alpha Beta Rho) est le standard absolu de l'industrie pour les marchés de taux d'intérêt et de devises. Contrairement au modèle de Heston qui modélise le prix de l'actif au comptant, le modèle SABR modélise directement le taux forward (ou prix forward).

## 1. Équations Fondamentales du Modèle SABR

Le système est régi par deux équations différentielles stochastiques (EDS) couplées :

**Dynamique du Forward Rate :**
$$dF_t = \alpha_t F_t^\beta dW_t^F$$

**Dynamique de la Volatilité :**
$$d\alpha_t = \nu \alpha_t dW_t^\alpha$$

**Corrélation entre les mouvements browniens :**
$$dW_t^F \cdot dW_t^\alpha = \rho dt$$

### Rôle du paramètre d'élasticité $\beta$
Le paramètre d'élasticité $\beta \in [0, 1]$ est la grande force du SABR, car il permet d'ajuster la forme de la distribution :
* Si $\beta = 0$ : Le modèle se comporte comme un modèle log-normal (volatilité indépendante de $F$).
* Si $\beta = 0.5$ : On retrouve une dynamique de type CIR (similaire à Heston).
* Si $\beta = 1$ : Le modèle devient strictement normal.

## 2. La Formule de Hagan (Approximation de la Volatilité Implicite)

La particularité mathématique majeure du modèle SABR est qu'il **n'admet aucune solution analytique exacte** pour le pricing d'options. 

Pour pallier ce problème, Patrick Hagan et son équipe (2002) ont développé un développement asymptotique extrêmement précis. Cette formule permet de calculer la volatilité implicite de Black (ou de Bachelier) $\sigma_B(K, F)$ de manière quasi-instantanée. C'est cette rapidité de calcul, combinée à une grande flexibilité pour ajuster le "smile" de volatilité, qui rend le SABR incontournable sur les salles de marché pour la calibration en temps réel.

## 3. Comparaison Structurelle : SABR vs Heston

Bien qu'ils soient tous deux des modèles de volatilité stochastique, Heston et SABR répondent à des logiques et des contraintes de marché différentes.

| Caractéristique | Modèle de Heston | Modèle SABR |
| :--- | :--- | :--- |
| **Dynamique de la volatilité** | Retour à la moyenne (processus CIR) | Log-normale (aucun retour à la moyenne) |
| **Positivité de la variance** | Conditionnée par l'inégalité de Feller ($2\kappa\theta \ge \sigma^2$) | Strictement garantie par la construction log-normale |
| **Méthode de Pricing** | Semi-analytique (Transformée de Fourier) | Approximation algébrique directe (Formule de Hagan) |
| **Marché de prédilection** | Actions (Equities) et Indices | Taux d'intérêt (Rates) et Devises (FX) |

*Note sur la dynamique :* L'absence de retour à la moyenne dans le modèle SABR implique que la volatilité peut mathématiquement dériver à l'infini sur le très long terme. Il est donc principalement calibré pour des horizons de temps plus courts que le modèle de Heston.