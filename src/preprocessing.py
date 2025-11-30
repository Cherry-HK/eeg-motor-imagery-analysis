"""
EEG Preprocessing Module for Motor Imagery Analysis

This module provides functions for loading, filtering, and cleaning EEG data
from the PhysioNet EEG Motor Movement/Imagery Dataset.
"""

import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def load_eeg_data(
    subject: int,
    runs: List[int],
    data_path: Optional[str] = None
) -> mne.io.Raw:
    """
    Load EEG data for a specific subject and runs.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int
        List of run numbers to load
    data_path : str, optional
        Path to store/load data. If None, uses default MNE data directory

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data
    """
    if data_path is None:
        data_path = str(Path.home() / 'mne_data')

    # Load raw data files
    raw_fnames = mne.datasets.eegbci.load_data(
        subjects=subject,
        runs=runs,
        path=data_path,
        update_path=True
    )

    # Read and concatenate raw files
    raw_list = [mne.io.read_raw_edf(fname, preload=True, verbose=False)
                for fname in raw_fnames]
    raw = mne.concatenate_raws(raw_list)

    # Standardize channel names for EEGBCI dataset
    mne.datasets.eegbci.standardize(raw)

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')

    return raw


def apply_filtering(
    raw: mne.io.Raw,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    notch_freq: Optional[float] = 60.0
) -> mne.io.Raw:
    """
    Apply bandpass and notch filtering to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency in Hz (default: 7 Hz for Mu band)
    h_freq : float
        High cutoff frequency in Hz (default: 30 Hz for Beta band)
    notch_freq : float, optional
        Frequency for notch filter in Hz (default: 60 Hz for US power line)
        Set to None to skip notch filtering

    Returns
    -------
    raw : mne.io.Raw
        Filtered raw EEG data
    """
    # Apply bandpass filter
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        verbose=False
    )

    # Apply notch filter if specified
    if notch_freq is not None:
        raw.notch_filter(
            freqs=notch_freq,
            verbose=False
        )

    return raw


def remove_artifacts_ica(
    raw: mne.io.Raw,
    n_components: int = 15,
    random_state: int = 42
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """
    Remove artifacts using Independent Component Analysis (ICA).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    n_components : int
        Number of ICA components (default: 15)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    raw : mne.io.Raw
        Cleaned raw EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    # Create and fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter='auto'
    )

    ica.fit(raw, verbose=False)

    # Automatically detect and exclude eye blink/movement components
    # EOG channels may not be present, so we use frontal channels as proxy
    eog_indices, eog_scores = ica.find_bads_eog(
        raw,
        ch_name=['Fp1', 'Fp2'],
        threshold=3.0,
        verbose=False
    )

    ica.exclude = eog_indices

    # Apply ICA to remove artifacts
    raw = ica.apply(raw, verbose=False)

    return raw, ica


def create_epochs(
    raw: mne.io.Raw,
    event_id: dict,
    tmin: float = -1.0,
    tmax: float = 4.0,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    reject: Optional[dict] = None
) -> mne.Epochs:
    """
    Create epochs from raw EEG data based on events.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    event_id : dict
        Dictionary mapping event names to event codes
        Example: {'left_fist': 1, 'right_fist': 2}
    tmin : float
        Start time before event in seconds (default: -1.0)
    tmax : float
        End time after event in seconds (default: 4.0)
    baseline : tuple of float
        Baseline correction interval (default: (-1.0, 0.0))
    reject : dict, optional
        Rejection parameters for bad epochs
        Example: {'eeg': 100e-6}

    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data
    """
    # Extract events from annotations
    events, _ = mne.events_from_annotations(raw, verbose=False)

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False
    )

    return epochs


def preprocess_pipeline(
    subject: int,
    runs: List[int],
    event_id: dict,
    data_path: Optional[str] = None,
    apply_ica: bool = False,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    notch_freq: Optional[float] = 60.0,
    tmin: float = -1.0,
    tmax: float = 4.0,
    baseline: Tuple[float, float] = (-1.0, 0.0),
    reject: Optional[dict] = None,
    verbose: bool = True
) -> Tuple[mne.Epochs, mne.io.Raw]:
    """
    Complete preprocessing pipeline for EEG motor imagery data.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int
        List of run numbers to load
    event_id : dict
        Dictionary mapping event names to event codes
    data_path : str, optional
        Path to store/load data
    apply_ica : bool
        Whether to apply ICA for artifact removal (default: False)
    l_freq : float
        Low cutoff frequency in Hz
    h_freq : float
        High cutoff frequency in Hz
    notch_freq : float, optional
        Frequency for notch filter in Hz
    tmin : float
        Start time before event in seconds
    tmax : float
        End time after event in seconds
    baseline : tuple of float
        Baseline correction interval
    reject : dict, optional
        Rejection parameters for bad epochs
    verbose : bool
        Whether to print progress information

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched data
    raw : mne.io.Raw
        Preprocessed raw data (for visualization)
    """
    if verbose:
        print(f"Loading data for subject {subject}, runs {runs}...")

    # Load data
    raw = load_eeg_data(subject, runs, data_path)

    if verbose:
        print(f"Applying filtering (bandpass: {l_freq}-{h_freq} Hz)...")

    # Apply filtering
    raw = apply_filtering(raw, l_freq, h_freq, notch_freq)

    # Optional ICA artifact removal
    if apply_ica:
        if verbose:
            print("Applying ICA for artifact removal...")
        raw, _ = remove_artifacts_ica(raw)

    if verbose:
        print("Creating epochs...")

    # Create epochs
    epochs = create_epochs(raw, event_id, tmin, tmax, baseline, reject)

    if verbose:
        print(f"Preprocessing complete. {len(epochs)} epochs created.")

    return epochs, raw


def get_motor_imagery_event_dict(task_type: str = 'left_right_fist') -> dict:
    """
    Get event dictionary for common motor imagery tasks.

    Parameters
    ----------
    task_type : str
        Type of motor imagery task:
        - 'left_right_fist': Left vs Right fist (imagined or real)
        - 'fists_feet': Both fists vs Both feet (imagined or real)

    Returns
    -------
    event_id : dict
        Event dictionary mapping task names to event codes
    """
    # Standard event codes from EEGBCI dataset
    # T0: Rest
    # T1: Left fist (or both fists)
    # T2: Right fist (or both feet)

    if task_type == 'left_right_fist':
        event_id = {
            'left_fist': 1,
            'right_fist': 2
        }
    elif task_type == 'fists_feet':
        event_id = {
            'both_fists': 1,
            'both_feet': 2
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return event_id


def get_recommended_runs(task_type: str = 'imagery_left_right') -> List[int]:
    """
    Get recommended run numbers for different task types.

    Parameters
    ----------
    task_type : str
        Type of task:
        - 'imagery_left_right': Imagined left/right fist movement
        - 'imagery_fists_feet': Imagined both fists/feet movement
        - 'real_left_right': Real left/right fist movement
        - 'real_fists_feet': Real both fists/feet movement

    Returns
    -------
    runs : list of int
        Recommended run numbers
    """
    run_mapping = {
        'imagery_left_right': [4, 8, 12],
        'imagery_fists_feet': [6, 10, 14],
        'real_left_right': [3, 7, 11],
        'real_fists_feet': [5, 9, 13]
    }

    if task_type not in run_mapping:
        raise ValueError(f"Unknown task type: {task_type}")

    return run_mapping[task_type]
