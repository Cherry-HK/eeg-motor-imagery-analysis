"""
Feature Extraction Module for EEG Motor Imagery Analysis

This module provides functions for extracting discriminative features from EEG data,
including Common Spatial Patterns (CSP) and Power Spectral Density (PSD).
"""

import mne
import numpy as np
from mne.decoding import CSP
from typing import Tuple, Optional, List


def extract_csp_features(
    epochs: mne.Epochs,
    n_components: int = 4,
    reg: Optional[str] = None,
    log: bool = True,
    norm_trace: bool = False
) -> Tuple[CSP, np.ndarray]:
    """
    Extract Common Spatial Pattern (CSP) features from epochs.

    CSP is a spatial filtering method that maximizes the variance of one class
    while minimizing it for another, making it ideal for motor imagery classification.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data with at least 2 classes
    n_components : int
        Number of CSP components to use (default: 4)
        Total features will be 2 * n_components
    reg : str, optional
        Regularization method ('shrinkage', 'ledoit_wolf', or None)
    log : bool
        Whether to apply log transformation to features (default: True)
    norm_trace : bool
        Whether to normalize the trace of the covariance matrices (default: False)

    Returns
    -------
    csp : mne.decoding.CSP
        Fitted CSP transformer
    features : np.ndarray
        CSP features, shape (n_epochs, n_components * 2)
    """
    # Initialize CSP
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=log,
        norm_trace=norm_trace
    )

    # Get data and labels
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # shape: (n_epochs,)

    # Fit CSP and transform data
    features = csp.fit_transform(X, y)

    return csp, features


def compute_band_power(
    epochs: mne.Epochs,
    freq_bands: dict = None,
    method: str = 'welch',
    normalize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute power spectral density in specified frequency bands.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    freq_bands : dict, optional
        Dictionary of frequency bands {name: (fmin, fmax)}
        Default: {'mu': (8, 13), 'beta': (13, 30)}
    method : str
        Method for PSD computation ('welch' or 'multitaper')
    normalize : bool
        Whether to normalize power by total power (default: True)

    Returns
    -------
    band_power : np.ndarray
        Power in each band, shape (n_epochs, n_channels, n_bands)
    band_names : list of str
        Names of frequency bands
    """
    if freq_bands is None:
        freq_bands = {
            'mu': (8, 13),
            'beta': (13, 30)
        }

    # Compute PSD
    spectrum = epochs.compute_psd(method=method, fmin=1, fmax=40, verbose=False)

    # Extract power in each band
    band_power_list = []
    band_names = []

    for band_name, (fmin, fmax) in freq_bands.items():
        # Get power in frequency band
        power = spectrum.get_data(fmin=fmin, fmax=fmax).mean(axis=-1)
        band_power_list.append(power)
        band_names.append(band_name)

    # Stack into array: (n_epochs, n_channels, n_bands)
    band_power = np.stack(band_power_list, axis=-1)

    # Normalize by total power if requested
    if normalize:
        total_power = spectrum.get_data().mean(axis=-1, keepdims=True)
        band_power = band_power / (total_power + 1e-10)

    return band_power, band_names


def compute_psd_features(
    epochs: mne.Epochs,
    freq_bands: dict = None,
    channels: Optional[List[str]] = None,
    method: str = 'welch',
    normalize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute PSD-based features for classification.

    This function extracts power in specific frequency bands from selected channels
    and flattens them into a feature vector.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    freq_bands : dict, optional
        Dictionary of frequency bands {name: (fmin, fmax)}
    channels : list of str, optional
        List of channel names to use. If None, uses motor cortex channels (C3, Cz, C4)
    method : str
        Method for PSD computation ('welch' or 'multitaper')
    normalize : bool
        Whether to normalize power by total power

    Returns
    -------
    features : np.ndarray
        PSD features, shape (n_epochs, n_channels * n_bands)
    feature_names : list of str
        Names of features
    """
    # Use motor cortex channels by default
    if channels is None:
        channels = ['C3', 'Cz', 'C4']

    # Select channels
    epochs_subset = epochs.copy().pick_channels(channels)

    # Compute band power
    band_power, band_names = compute_band_power(
        epochs_subset,
        freq_bands=freq_bands,
        method=method,
        normalize=normalize
    )

    # Flatten features: (n_epochs, n_channels * n_bands)
    n_epochs, n_channels, n_bands = band_power.shape
    features = band_power.reshape(n_epochs, n_channels * n_bands)

    # Create feature names
    feature_names = []
    for ch in channels:
        for band in band_names:
            feature_names.append(f"{ch}_{band}")

    return features, feature_names


def compute_erds_map(
    epochs: mne.Epochs,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    mode: str = 'percent'
) -> mne.time_frequency.EpochsTFR:
    """
    Compute Event-Related Desynchronization/Synchronization (ERD/ERS) map.

    ERD represents a decrease in oscillatory power during motor imagery,
    while ERS represents an increase.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    baseline : tuple of float
        Baseline period for normalization (default: (-1.0, 0.0))
    mode : str
        Type of baseline correction ('percent', 'ratio', 'zscore', 'mean', 'logratio')

    Returns
    -------
    tfr : mne.time_frequency.EpochsTFR
        Time-frequency representation with baseline correction
    """
    # Compute time-frequency representation using multitaper method
    freqs = np.arange(7, 31, 1)  # 7-30 Hz (Mu and Beta bands)
    n_cycles = freqs / 2.0  # Number of cycles increases with frequency

    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        verbose=False
    )

    # Apply baseline correction
    tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

    return tfr


def extract_motor_cortex_features(
    epochs: mne.Epochs,
    feature_type: str = 'csp',
    **kwargs
) -> Tuple[np.ndarray, object]:
    """
    Extract features from motor cortex regions for motor imagery classification.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    feature_type : str
        Type of features to extract:
        - 'csp': Common Spatial Patterns
        - 'psd': Power Spectral Density
        - 'combined': Both CSP and PSD features
    **kwargs : dict
        Additional arguments passed to feature extraction functions

    Returns
    -------
    features : np.ndarray
        Extracted features
    transformer : object
        Fitted transformer (CSP object or None for PSD)
    """
    if feature_type == 'csp':
        # Extract CSP features
        n_components = kwargs.get('n_components', 4)
        reg = kwargs.get('reg', None)
        log = kwargs.get('log', True)

        csp, features = extract_csp_features(
            epochs,
            n_components=n_components,
            reg=reg,
            log=log
        )
        return features, csp

    elif feature_type == 'psd':
        # Extract PSD features
        channels = kwargs.get('channels', ['C3', 'Cz', 'C4'])
        freq_bands = kwargs.get('freq_bands', None)
        method = kwargs.get('method', 'welch')
        normalize = kwargs.get('normalize', True)

        features, _ = compute_psd_features(
            epochs,
            freq_bands=freq_bands,
            channels=channels,
            method=method,
            normalize=normalize
        )
        return features, None

    elif feature_type == 'combined':
        # Extract both CSP and PSD features
        csp, csp_features = extract_csp_features(epochs, **kwargs)
        psd_features, _ = compute_psd_features(epochs, **kwargs)

        # Concatenate features
        features = np.hstack([csp_features, psd_features])
        return features, csp

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def get_channel_importance(
    csp: CSP,
    channel_names: List[str]
) -> Tuple[np.ndarray, List[int]]:
    """
    Get channel importance from fitted CSP filters.

    Parameters
    ----------
    csp : mne.decoding.CSP
        Fitted CSP transformer
    channel_names : list of str
        Names of EEG channels

    Returns
    -------
    importance : np.ndarray
        Importance score for each channel
    top_channels : list of int
        Indices of most important channels
    """
    # Get CSP patterns (inverse of filters)
    patterns = csp.patterns_

    # Compute importance as absolute mean across components
    importance = np.abs(patterns).mean(axis=0)

    # Get indices of top channels
    top_channels = np.argsort(importance)[::-1]

    return importance, top_channels.tolist()


def compute_lateralization_index(
    epochs: mne.Epochs,
    freq_band: Tuple[float, float] = (8, 13)
) -> np.ndarray:
    """
    Compute lateralization index for motor cortex.

    The lateralization index quantifies the difference in power between
    left (C3) and right (C4) motor cortex.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    freq_band : tuple of float
        Frequency band for power computation (default: (8, 13) for Mu band)

    Returns
    -------
    li : np.ndarray
        Lateralization index for each epoch, shape (n_epochs,)
        Positive values indicate right hemisphere dominance
        Negative values indicate left hemisphere dominance
    """
    # Select C3 and C4 channels
    try:
        epochs_c3c4 = epochs.copy().pick_channels(['C3', 'C4'])
    except ValueError:
        raise ValueError("Epochs must contain C3 and C4 channels")

    # Compute PSD
    spectrum = epochs_c3c4.compute_psd(
        method='welch',
        fmin=freq_band[0],
        fmax=freq_band[1],
        verbose=False
    )

    # Get power for each channel
    power = spectrum.get_data().mean(axis=-1)  # Average over frequencies
    power_c3 = power[:, 0]
    power_c4 = power[:, 1]

    # Compute lateralization index: (C4 - C3) / (C4 + C3)
    li = (power_c4 - power_c3) / (power_c4 + power_c3 + 1e-10)

    return li
