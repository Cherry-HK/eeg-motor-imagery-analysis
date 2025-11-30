"""
Visualization Module for EEG Motor Imagery Analysis

This module provides functions for visualizing EEG data, features,
and classification results.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from mne.decoding import CSP


def plot_raw_psd(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 40.0,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot power spectral density of raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    fmin : float
        Minimum frequency to plot
    fmax : float
        Maximum frequency to plot
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig = raw.compute_psd(fmin=fmin, fmax=fmax).plot(
        picks='eeg',
        show=show
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PSD plot saved to {save_path}")

    return fig


def plot_epochs_image(
    epochs: mne.Epochs,
    picks: Optional[Union[str, List[str]]] = None,
    combine: str = 'mean',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> List[plt.Figure]:
    """
    Plot epochs as an image (heatmap over time).

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    picks : str or list of str, optional
        Channels to plot
    combine : str
        How to combine channels ('mean', 'median', None)
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figure objects
    """
    if picks is None:
        picks = ['C3', 'Cz', 'C4']

    figs = epochs.plot_image(
        picks=picks,
        combine=combine,
        show=show
    )

    if save_path is not None:
        if isinstance(figs, list):
            for i, fig in enumerate(figs):
                path = Path(save_path).parent / f"{Path(save_path).stem}_{i}{Path(save_path).suffix}"
                fig.savefig(path, dpi=300, bbox_inches='tight')
        else:
            figs.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Epochs image saved to {save_path}")

    return figs


def plot_erds_map(
    epochs: mne.Epochs,
    picks: Optional[List[str]] = None,
    fmin: float = 7.0,
    fmax: float = 30.0,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    mode: str = 'percent',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Event-Related Desynchronization/Synchronization (ERD/ERS) map.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    picks : list of str, optional
        Channels to plot (default: motor cortex channels)
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    baseline : tuple of float
        Baseline period
    mode : str
        Baseline correction mode
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if picks is None:
        picks = ['C3', 'Cz', 'C4']

    # Compute time-frequency representation
    freqs = np.arange(fmin, fmax + 1, 1)
    n_cycles = freqs / 2.0

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        picks=picks,
        verbose=False
    )

    # Apply baseline
    power = power.apply_baseline(baseline=baseline, mode=mode)

    # Plot
    fig = power.plot(
        picks=picks,
        baseline=baseline,
        mode=mode,
        show=show,
        combine='mean'
    )[0]

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ERD/ERS map saved to {save_path}")

    return fig


def plot_csp_patterns(
    csp: CSP,
    info: mne.Info,
    n_components: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot CSP spatial patterns on a topographic map.

    Parameters
    ----------
    csp : mne.decoding.CSP
        Fitted CSP object
    info : mne.Info
        MNE info object containing channel locations
    n_components : int
        Number of components to plot
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get CSP patterns
    patterns = csp.patterns_

    # Create evoked object for plotting
    n_components = min(n_components, patterns.shape[0])

    fig, axes = plt.subplots(2, n_components // 2, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(n_components):
        # Create evoked object with pattern as data
        evoked = mne.EvokedArray(
            patterns[i:i+1].T,
            info,
            tmin=0
        )

        # Plot topomap
        mne.viz.plot_topomap(
            evoked.data[:, 0],
            evoked.info,
            axes=axes[i],
            show=False
        )
        axes[i].set_title(f'Component {i+1}')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CSP patterns saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list of str, optional
        Class labels
    normalize : bool
        Whether to normalize the confusion matrix
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use default labels if not provided
    if labels is None:
        labels = np.unique(y_true)

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get unique classes
    classes = np.unique(y_true)

    if len(classes) != 2:
        raise ValueError("ROC curve is only available for binary classification")

    # Compute ROC curve (specify pos_label for non-standard labels)
    pos_label = classes[1]  # Use the second class as positive
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Chance level')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_cv_scores(
    scores: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cross-validation scores.

    Parameters
    ----------
    scores : dict
        Dictionary containing cross-validation scores
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric_name, metric_scores in scores.items():
        if metric_name.startswith('test_'):
            metric = metric_name.replace('test_', '')
            ax.plot(metric_scores, marker='o', label=metric.upper())

    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Scores')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV scores plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_classifier_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of different classifiers.

    Parameters
    ----------
    results : dict
        Dictionary mapping classifier names to their performance metrics
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    classifiers = list(results.keys())
    mean_scores = [results[clf]['mean_accuracy'] for clf in classifiers]
    std_scores = [results[clf]['std_accuracy'] for clf in classifiers]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(classifiers))
    bars = ax.bar(x_pos, mean_scores, yerr=std_scores, capsize=5,
                   color='steelblue', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Performance Comparison')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classifier comparison saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_lateralization_index(
    epochs: mne.Epochs,
    event_dict: Dict[str, int],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot lateralization index over time for different conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    event_dict : dict
        Dictionary mapping event names to codes
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from features import compute_lateralization_index

    fig, ax = plt.subplots(figsize=(10, 6))

    for event_name, event_code in event_dict.items():
        # Select epochs for this condition
        epochs_subset = epochs[event_name]

        # Compute lateralization index
        li = compute_lateralization_index(epochs_subset)

        # Plot
        time_points = epochs.times
        li_mean = li.mean()
        li_std = li.std()

        ax.axhline(li_mean, label=f'{event_name} (mean={li_mean:.3f})',
                   linestyle='--', linewidth=2)
        ax.fill_between([time_points[0], time_points[-1]],
                        li_mean - li_std, li_mean + li_std,
                        alpha=0.2)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lateralization Index')
    ax.set_title('Motor Cortex Lateralization (C4-C3)/(C4+C3)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lateralization index plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_channel_importance(
    csp: CSP,
    channel_names: List[str],
    n_top: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot channel importance from CSP filters.

    Parameters
    ----------
    csp : mne.decoding.CSP
        Fitted CSP object
    channel_names : list of str
        Names of EEG channels
    n_top : int
        Number of top channels to display
    save_path : str or Path, optional
        Path to save the figure
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from features import get_channel_importance

    # Get importance scores
    importance, top_indices = get_channel_importance(csp, channel_names)

    # Select top channels
    top_indices = top_indices[:n_top]
    top_channels = [channel_names[i] for i in top_indices]
    top_importance = importance[top_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top_channels))
    ax.barh(y_pos, top_importance, color='steelblue', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_channels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {n_top} Most Important Channels (CSP)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Channel importance plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_analysis_report(
    epochs: mne.Epochs,
    csp: CSP,
    results: Dict,
    output_dir: Union[str, Path],
    subject_id: int
) -> None:
    """
    Create a comprehensive analysis report with all visualizations.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    csp : mne.decoding.CSP
        Fitted CSP object
    results : dict
        Classification results
    output_dir : str or Path
        Directory to save the report
    subject_id : int
        Subject identifier
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating analysis report for Subject {subject_id}...")

    # Plot CSP patterns
    plot_csp_patterns(
        csp,
        epochs.info,
        save_path=output_dir / f"subject_{subject_id}_csp_patterns.png",
        show=False
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        results['y_true'],
        results['y_pred'],
        save_path=output_dir / f"subject_{subject_id}_confusion_matrix.png",
        show=False
    )

    # Plot ROC curve (if binary classification)
    if len(np.unique(results['y_true'])) == 2:
        plot_roc_curve(
            results['y_true'],
            results['y_proba'][:, 1],
            save_path=output_dir / f"subject_{subject_id}_roc_curve.png",
            show=False
        )

    # Plot channel importance
    plot_channel_importance(
        csp,
        epochs.ch_names,
        save_path=output_dir / f"subject_{subject_id}_channel_importance.png",
        show=False
    )

    print(f"Report saved to {output_dir}")
