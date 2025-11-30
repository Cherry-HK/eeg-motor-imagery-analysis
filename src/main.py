"""
Main Pipeline Script for EEG Motor Imagery Classification

This script demonstrates the complete pipeline from data loading to classification
and visualization for the PhysioNet EEG Motor Movement/Imagery Dataset.
"""

import argparse
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom modules
from preprocessing import (
    preprocess_pipeline,
    get_motor_imagery_event_dict,
    get_recommended_runs
)
from features import extract_csp_features
from classification import (
    create_csp_lda_pipeline,
    train_and_evaluate,
    evaluate_on_test_set,
    compare_classifiers,
    save_model
)
from visualization import (
    plot_raw_psd,
    plot_csp_patterns,
    plot_confusion_matrix,
    plot_roc_curve,
    create_analysis_report
)


def run_single_subject_analysis(
    subject: int,
    task_type: str = 'imagery_left_right',
    apply_ica: bool = False,
    output_dir: str = '../outputs',
    data_path: str = None,
    verbose: bool = True
):
    """
    Run complete analysis for a single subject.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    task_type : str
        Type of motor imagery task
    apply_ica : bool
        Whether to apply ICA for artifact removal
    output_dir : str
        Directory to save outputs
    data_path : str, optional
        Path to data directory
    verbose : bool
        Whether to print progress
    """
    if verbose:
        print("=" * 70)
        print(f"RUNNING ANALYSIS FOR SUBJECT {subject}")
        print("=" * 70)

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get recommended runs for task type
    runs = get_recommended_runs(task_type)

    # Get event dictionary
    if 'left_right' in task_type:
        event_id = get_motor_imagery_event_dict('left_right_fist')
    else:
        event_id = get_motor_imagery_event_dict('fists_feet')

    if verbose:
        print(f"\nTask Type: {task_type}")
        print(f"Runs: {runs}")
        print(f"Events: {event_id}")

    # Step 1: Preprocessing
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 1: PREPROCESSING")
        print("-" * 70)

    epochs, raw = preprocess_pipeline(
        subject=subject,
        runs=runs,
        event_id=event_id,
        data_path=data_path,
        apply_ica=apply_ica,
        l_freq=7.0,
        h_freq=30.0,
        notch_freq=60.0,
        verbose=verbose
    )

    # Visualize PSD
    if verbose:
        print("\nGenerating PSD plot...")
    plot_raw_psd(
        raw,
        fmin=0.5,
        fmax=40.0,
        save_path=output_dir / f"subject_{subject}_psd.png",
        show=False
    )

    # Step 2: Classification
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 2: CLASSIFICATION")
        print("-" * 70)

    # Create pipeline
    pipeline = create_csp_lda_pipeline(n_components=4)

    # Cross-validation
    if verbose:
        print("\nPerforming cross-validation...")

    scores = train_and_evaluate(
        epochs,
        pipeline,
        cv=5,
        scoring='accuracy',
        verbose=verbose
    )

    # Step 3: Train final model and evaluate
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 3: FINAL MODEL TRAINING")
        print("-" * 70)

    # Split data for final evaluation
    n_epochs = len(epochs)
    train_size = int(0.8 * n_epochs)

    # Get indices
    indices = np.arange(n_epochs)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Split epochs
    train_epochs = epochs[train_indices]
    test_epochs = epochs[test_indices]

    # Train and evaluate
    results = evaluate_on_test_set(
        train_epochs,
        test_epochs,
        pipeline,
        verbose=verbose
    )

    # Step 4: Visualization
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("-" * 70)

    # Get fitted CSP from pipeline
    X_train = train_epochs.get_data()
    y_train = train_epochs.events[:, -1]
    pipeline.fit(X_train, y_train)
    csp = pipeline.named_steps['CSP']

    # Create comprehensive report
    create_analysis_report(
        epochs,
        csp,
        results,
        output_dir,
        subject
    )

    # Save model
    model_path = output_dir / f"subject_{subject}_model.pkl"
    save_model(
        pipeline,
        model_path,
        metadata={
            'subject': subject,
            'task_type': task_type,
            'accuracy': results['accuracy'],
            'roc_auc': results['roc_auc']
        }
    )

    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"Results saved to: {output_dir}")

    return results, pipeline, epochs


def run_multi_subject_analysis(
    subjects: list,
    task_type: str = 'imagery_left_right',
    apply_ica: bool = False,
    output_dir: str = '../outputs',
    data_path: str = None,
    verbose: bool = True
):
    """
    Run analysis across multiple subjects.

    Parameters
    ----------
    subjects : list of int
        List of subject numbers
    task_type : str
        Type of motor imagery task
    apply_ica : bool
        Whether to apply ICA
    output_dir : str
        Directory to save outputs
    data_path : str, optional
        Path to data directory
    verbose : bool
        Whether to print progress
    """
    results_all = {}

    for subject in subjects:
        try:
            results, pipeline, epochs = run_single_subject_analysis(
                subject=subject,
                task_type=task_type,
                apply_ica=apply_ica,
                output_dir=output_dir,
                data_path=data_path,
                verbose=verbose
            )

            results_all[subject] = {
                'accuracy': results['accuracy'],
                'roc_auc': results['roc_auc']
            }

        except Exception as e:
            print(f"\nError processing subject {subject}: {str(e)}")
            continue

    # Print summary
    if verbose and results_all:
        print("\n" + "=" * 70)
        print("MULTI-SUBJECT SUMMARY")
        print("=" * 70)

        accuracies = [r['accuracy'] for r in results_all.values()]
        roc_aucs = [r['roc_auc'] for r in results_all.values()]

        print(f"\nNumber of subjects: {len(results_all)}")
        print(f"Mean Accuracy: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")
        print(f"Mean ROC AUC: {np.mean(roc_aucs):.3f} (+/- {np.std(roc_aucs):.3f})")

        print("\nPer-subject results:")
        for subject, result in results_all.items():
            print(f"  Subject {subject:3d}: Accuracy={result['accuracy']:.3f}, "
                  f"ROC AUC={result['roc_auc']:.3f}")

    return results_all


def run_classifier_comparison(
    subject: int,
    task_type: str = 'imagery_left_right',
    output_dir: str = '../outputs',
    data_path: str = None,
    verbose: bool = True
):
    """
    Compare different classifiers on a single subject.

    Parameters
    ----------
    subject : int
        Subject number
    task_type : str
        Type of motor imagery task
    output_dir : str
        Directory to save outputs
    data_path : str, optional
        Path to data directory
    verbose : bool
        Whether to print progress
    """
    if verbose:
        print("=" * 70)
        print(f"COMPARING CLASSIFIERS FOR SUBJECT {subject}")
        print("=" * 70)

    # Get data
    runs = get_recommended_runs(task_type)

    if 'left_right' in task_type:
        event_id = get_motor_imagery_event_dict('left_right_fist')
    else:
        event_id = get_motor_imagery_event_dict('fists_feet')

    # Preprocess
    epochs, _ = preprocess_pipeline(
        subject=subject,
        runs=runs,
        event_id=event_id,
        data_path=data_path,
        apply_ica=False,
        verbose=verbose
    )

    # Compare classifiers
    results = compare_classifiers(epochs, cv=5, verbose=verbose)

    # Visualize comparison
    from visualization import plot_classifier_comparison

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_classifier_comparison(
        results,
        save_path=output_dir / f"subject_{subject}_classifier_comparison.png",
        show=False
    )

    return results


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='EEG Motor Imagery Classification Pipeline'
    )

    parser.add_argument(
        '--subject',
        type=int,
        default=1,
        help='Subject number (1-109)'
    )

    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        help='Multiple subject numbers for multi-subject analysis'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='imagery_left_right',
        choices=['imagery_left_right', 'imagery_fists_feet',
                 'real_left_right', 'real_fists_feet'],
        help='Type of motor imagery task'
    )

    parser.add_argument(
        '--ica',
        action='store_true',
        help='Apply ICA for artifact removal'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../outputs',
        help='Output directory for results'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to data directory'
    )

    parser.add_argument(
        '--compare-classifiers',
        action='store_true',
        help='Compare different classifiers'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Run appropriate analysis
    if args.compare_classifiers:
        run_classifier_comparison(
            subject=args.subject,
            task_type=args.task,
            output_dir=args.output,
            data_path=args.data_path,
            verbose=verbose
        )
    elif args.subjects:
        run_multi_subject_analysis(
            subjects=args.subjects,
            task_type=args.task,
            apply_ica=args.ica,
            output_dir=args.output,
            data_path=args.data_path,
            verbose=verbose
        )
    else:
        run_single_subject_analysis(
            subject=args.subject,
            task_type=args.task,
            apply_ica=args.ica,
            output_dir=args.output,
            data_path=args.data_path,
            verbose=verbose
        )


if __name__ == '__main__':
    main()
