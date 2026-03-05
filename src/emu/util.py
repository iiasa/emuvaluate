import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def detrend_gaussian(df, time_col="time", tau=40):
    """
    Detrend each column in df using Gaussian smoothing.
    """
    var_cols = [c for c in df.columns if c != time_col]
    sigma = tau  # tau already in time units (months)

    df_trend = df.copy()
    for c in var_cols:
        df_trend[c] = gaussian_filter1d(df[c].values, sigma=sigma, mode="nearest")

    df_detrended = df.copy()
    df_detrended[var_cols] = df[var_cols] - df_trend[var_cols]

    return df_detrended, df_trend


def weights_calculate(x0, X, tau):
    """Gaussian kernel weights for local regression."""
    return np.exp(-np.sum((X - x0) ** 2, axis=1) / (2 * tau**2))


def local_weighted_regression(x0, X, Y, tau):
    """Locally weighted linear regression at point x0."""
    # ensure x0 is scalar
    x0 = float(np.squeeze(x0))

    X = np.c_[np.ones(len(X)), X]  # add intercept
    x0_vec = np.array([1, x0])
    W = np.diag(weights_calculate(np.array([[x0]]), X[:, 1:], tau))
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ Y)
    return float(x0_vec @ theta)


def detrend_dataframe(df, time_col="time", tau=20, verbose=True):
    """
    Detrend each column in df using local weighted regression (Gaussian kernel).
    """
    time_vals = df[time_col].to_numpy().reshape(-1, 1)
    variable_cols = [c for c in df.columns if c != time_col]

    df_trend = pd.DataFrame({time_col: df[time_col]})
    df_detrended = pd.DataFrame({time_col: df[time_col]})

    for col in variable_cols:
        Y = df[col].to_numpy()
        if verbose:
            print(f"Detrending {col} ...", end="\r")
        trend_vals = np.array(
            [local_weighted_regression(x0, time_vals, Y, tau) for x0 in time_vals]
        )
        df_trend[col] = trend_vals
        df_detrended[col] = Y - trend_vals

    if verbose:
        print(f"\n✅ Detrending complete. tau={tau}")
    return df_detrended, df_trend



def deseasonalise_dataframe(df, time_col="time", period=12):
    """
    Remove a fixed mean seasonal cycle.
    Assumes time_col is integer-like so: season = time % period.
    """
    variable_cols = [c for c in df.columns if c != time_col]

    # seasonal index
    season_index = df[time_col].astype(int) % period

    # compute climatology
    climatology = df.groupby(season_index)[variable_cols].mean()

    # expand climatology to full series
    seasonal = climatology.iloc[season_index].reset_index(drop=True)

    # subtract
    deseasonalised = df.copy()
    deseasonalised[variable_cols] = df[variable_cols].values - seasonal.values

    # attach time column to seasonal
    seasonal.insert(0, time_col, df[time_col].values)

    return deseasonalised, seasonal

