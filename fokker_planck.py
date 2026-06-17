import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simpson  # corrected import
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def compute_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Use first principal component as composite factor
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def fokker_planck_density(returns, grid_points=50):
    """Estimate probability density via KDE and discretise on a grid."""
    if len(returns) < 5:
        return None, None
    kde = gaussian_kde(returns)
    x_grid = np.linspace(returns.min() - 0.5, returns.max() + 0.5, grid_points)
    density = kde.evaluate(x_grid)
    # Normalise to integrate to 1
    density = density / simpson(density, x_grid)
    return x_grid, density

def entropy_production_rate(returns, macro_df, grid_points=50):
    """
    Compute entropy production rate of the density evolution under Fokker-Planck.
    Approximate: split the window into two halves, estimate densities, compute KL divergence.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.0
    # Compute macro factor for the window
    macro_factor = compute_macro_factor(macro_df)
    # Split window into two halves
    split = len(returns) // 2
    ret_first = returns[:split]
    ret_second = returns[split:]
    if len(ret_first) < 5 or len(ret_second) < 5:
        return 0.0
    # Estimate densities for each half
    x_grid1, density1 = fokker_planck_density(ret_first, grid_points)
    x_grid2, density2 = fokker_planck_density(ret_second, grid_points)
    if density1 is None or density2 is None:
        return 0.0
    # Interpolate density2 onto x_grid1 for KL divergence
    from scipy.interpolate import interp1d
    f2 = interp1d(x_grid2, density2, kind='linear', fill_value=0.0, bounds_error=False)
    density2_on_grid1 = f2(x_grid1)
    # KL divergence: D_KL(p1 || p2)
    # Use only where both densities are positive
    mask = (density1 > 1e-12) & (density2_on_grid1 > 1e-12)
    if np.sum(mask) < 2:
        return 0.0
    kl = simpson(density1[mask] * np.log(density1[mask] / density2_on_grid1[mask]), x_grid1[mask])
    # Entropy production rate = KL / (time difference)
    # Time difference = split days (approximately half the window)
    dt = split / len(returns)
    entropy_rate = kl / (dt + 1e-8)
    return float(entropy_rate)
