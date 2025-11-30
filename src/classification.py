"""
Classification Module for EEG Motor Imagery Analysis

This module provides functions for training and evaluating classifiers
for motor imagery tasks using various machine learning algorithms.
"""

import numpy as np
import mne
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    ShuffleSplit,
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from mne.decoding import CSP
from typing import Dict, Tuple, Optional, List, Union
import pickle
from pathlib import Path


def create_csp_lda_pipeline(
    n_components: int = 4,
    reg: Optional[str] = None,
    solver: str = 'lsqr',
    shrinkage: str = 'auto'
) -> Pipeline:
    """
    Create a CSP + LDA classification pipeline.

    This is the standard approach for motor imagery classification.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : str, optional
        CSP regularization method
    solver : str
        LDA solver ('svd', 'lsqr', 'eigen')
    shrinkage : str or float
        LDA shrinkage parameter

    Returns
    -------
    pipeline : Pipeline
        Scikit-learn pipeline with CSP and LDA
    """
    pipeline = Pipeline([
        ('CSP', CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)),
        ('LDA', LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage))
    ])

    return pipeline


def create_csp_svm_pipeline(
    n_components: int = 4,
    reg: Optional[str] = None,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale'
) -> Pipeline:
    """
    Create a CSP + SVM classification pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : str, optional
        CSP regularization method
    kernel : str
        SVM kernel type ('linear', 'rbf', 'poly')
    C : float
        Regularization parameter
    gamma : str or float
        Kernel coefficient

    Returns
    -------
    pipeline : Pipeline
        Scikit-learn pipeline with CSP and SVM
    """
    pipeline = Pipeline([
        ('CSP', CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)),
        ('SVM', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
    ])

    return pipeline


def create_csp_rf_pipeline(
    n_components: int = 4,
    reg: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Create a CSP + Random Forest classification pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : str, optional
        CSP regularization method
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    random_state : int
        Random seed

    Returns
    -------
    pipeline : Pipeline
        Scikit-learn pipeline with CSP and Random Forest
    """
    pipeline = Pipeline([
        ('CSP', CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)),
        ('RF', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])

    return pipeline


def train_and_evaluate(
    epochs: mne.Epochs,
    pipeline: Pipeline,
    cv: Union[int, object] = 5,
    scoring: Union[str, List[str]] = 'accuracy',
    return_train_score: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Train and evaluate a classification pipeline using cross-validation.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    pipeline : Pipeline
        Scikit-learn pipeline
    cv : int or cross-validation object
        Cross-validation strategy (default: 5-fold)
    scoring : str or list of str
        Scoring metric(s) ('accuracy', 'roc_auc', etc.)
    return_train_score : bool
        Whether to return training scores
    verbose : bool
        Whether to print results

    Returns
    -------
    scores : dict
        Dictionary containing cross-validation scores
    """
    # Get data and labels
    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Perform cross-validation
    if isinstance(scoring, str):
        scores_array = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            verbose=0
        )
        scores = {
            f'test_{scoring}': scores_array
        }
    else:
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            verbose=0
        )

    # Print results
    if verbose:
        print("\nCross-Validation Results:")
        print("-" * 50)
        for metric_name, metric_scores in scores.items():
            if metric_name.startswith('test_'):
                metric = metric_name.replace('test_', '')
                mean_score = np.mean(metric_scores)
                std_score = np.std(metric_scores)
                print(f"{metric.upper()}: {mean_score:.3f} (+/- {std_score:.3f})")

    return scores


def evaluate_on_test_set(
    train_epochs: mne.Epochs,
    test_epochs: mne.Epochs,
    pipeline: Pipeline,
    verbose: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Train on training set and evaluate on separate test set.

    Parameters
    ----------
    train_epochs : mne.Epochs
        Training epochs
    test_epochs : mne.Epochs
        Test epochs
    pipeline : Pipeline
        Scikit-learn pipeline
    verbose : bool
        Whether to print results

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Get training data
    X_train = train_epochs.get_data()
    y_train = train_epochs.events[:, -1]

    # Get test data
    X_test = test_epochs.get_data()
    y_test = test_epochs.events[:, -1]

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC for binary classification
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    # Print results
    if verbose:
        print("\nTest Set Evaluation:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return results


def cross_validate_multiple_subjects(
    subject_epochs: Dict[int, mne.Epochs],
    pipeline: Pipeline,
    cv: int = 5,
    scoring: str = 'accuracy',
    verbose: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Perform cross-validation across multiple subjects.

    Parameters
    ----------
    subject_epochs : dict
        Dictionary mapping subject IDs to their epochs
    pipeline : Pipeline
        Scikit-learn pipeline
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    verbose : bool
        Whether to print results

    Returns
    -------
    results : dict
        Dictionary mapping subject IDs to their results
    """
    results = {}

    for subject_id, epochs in subject_epochs.items():
        if verbose:
            print(f"\nEvaluating Subject {subject_id}...")

        scores = train_and_evaluate(
            epochs,
            pipeline,
            cv=cv,
            scoring=scoring,
            verbose=False
        )

        mean_score = np.mean(scores[f'test_{scoring}'])
        std_score = np.std(scores[f'test_{scoring}'])

        results[subject_id] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': scores[f'test_{scoring}']
        }

        if verbose:
            print(f"  {scoring.upper()}: {mean_score:.3f} (+/- {std_score:.3f})")

    # Calculate overall statistics
    if verbose:
        all_means = [r['mean_score'] for r in results.values()]
        overall_mean = np.mean(all_means)
        overall_std = np.std(all_means)
        print(f"\nOverall Performance Across Subjects:")
        print(f"  Mean {scoring.upper()}: {overall_mean:.3f} (+/- {overall_std:.3f})")

    return results


def perform_shuffle_split_cv(
    epochs: mne.Epochs,
    pipeline: Pipeline,
    n_splits: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform shuffle split cross-validation.

    This is particularly useful for small datasets where you want
    to evaluate stability across different train/test splits.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    pipeline : Pipeline
        Scikit-learn pipeline
    n_splits : int
        Number of splits
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
    verbose : bool
        Whether to print results

    Returns
    -------
    scores : dict
        Dictionary containing scores for each split
    """
    cv = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )

    scores = train_and_evaluate(
        epochs,
        pipeline,
        cv=cv,
        scoring='accuracy',
        verbose=False
    )

    if verbose:
        mean_score = np.mean(scores['test_accuracy'])
        std_score = np.std(scores['test_accuracy'])
        print(f"\nShuffle Split CV Results ({n_splits} splits):")
        print(f"Accuracy: {mean_score:.3f} (+/- {std_score:.3f})")

    return scores


def save_model(
    pipeline: Pipeline,
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    filepath : str or Path
        Path to save the model
    metadata : dict, optional
        Additional metadata to save with the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'pipeline': pipeline,
        'metadata': metadata or {}
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Tuple[Pipeline, Dict]:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model

    Returns
    -------
    pipeline : Pipeline
        Loaded pipeline
    metadata : dict
        Model metadata
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    pipeline = model_data['pipeline']
    metadata = model_data.get('metadata', {})

    print(f"Model loaded from {filepath}")

    return pipeline, metadata


def compare_classifiers(
    epochs: mne.Epochs,
    cv: int = 5,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of different classifiers.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results

    Returns
    -------
    results : dict
        Dictionary mapping classifier names to their performance
    """
    # Define classifiers to compare
    classifiers = {
        'CSP+LDA': create_csp_lda_pipeline(),
        'CSP+SVM(RBF)': create_csp_svm_pipeline(kernel='rbf'),
        'CSP+SVM(Linear)': create_csp_svm_pipeline(kernel='linear'),
        'CSP+RandomForest': create_csp_rf_pipeline()
    }

    results = {}

    if verbose:
        print("\nComparing Classifiers:")
        print("=" * 50)

    for name, pipeline in classifiers.items():
        scores = train_and_evaluate(
            epochs,
            pipeline,
            cv=cv,
            scoring='accuracy',
            verbose=False
        )

        mean_score = np.mean(scores['test_accuracy'])
        std_score = np.std(scores['test_accuracy'])

        results[name] = {
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
            'scores': scores['test_accuracy']
        }

        if verbose:
            print(f"{name:20s}: {mean_score:.3f} (+/- {std_score:.3f})")

    return results


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract feature importance from trained pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline
    feature_names : list of str, optional
        Names of features

    Returns
    -------
    importance : np.ndarray
        Feature importance scores
    """
    # Get the classifier from the pipeline
    classifier = pipeline.steps[-1][1]

    # Extract importance based on classifier type
    if hasattr(classifier, 'coef_'):
        # Linear models (LDA, LinearSVM)
        importance = np.abs(classifier.coef_).flatten()
    elif hasattr(classifier, 'feature_importances_'):
        # Tree-based models (RandomForest)
        importance = classifier.feature_importances_
    else:
        raise ValueError("Classifier does not support feature importance")

    return importance
