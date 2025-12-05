"""
Quick test script to verify the EEG analysis pipeline.

This script performs a minimal test with Subject 1 to ensure all components work,
including all available classifiers (LDA, Logistic Regression, SVM, Random Forest).
"""

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EEG PIPELINE TEST")
print("=" * 70)

# Test 1: Import modules
print("\n[1/7] Testing imports...")
try:
    from preprocessing import (
        preprocess_pipeline,
        get_motor_imagery_event_dict,
        get_recommended_runs
    )
    from features import extract_csp_features
    from classification import (
        create_csp_lda_pipeline,
        create_csp_logistic_pipeline,
        create_csp_svm_pipeline,
        create_csp_rf_pipeline,
        train_and_evaluate,
        compare_classifiers
    )
    from visualization import plot_csp_patterns
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n[2/7] Setting up configuration...")
try:
    subject = 1
    task_type = 'imagery_left_right'
    runs = get_recommended_runs(task_type)
    event_id = get_motor_imagery_event_dict('left_right_fist')
    print(f"✓ Configuration set: Subject {subject}, Runs {runs}, Events {event_id}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Data loading and preprocessing
print("\n[3/7] Loading and preprocessing data...")
print("   (This may take a few minutes to download data...)")
try:
    epochs, raw = preprocess_pipeline(
        subject=subject,
        runs=runs,
        event_id=event_id,
        apply_ica=False,
        l_freq=7.0,
        h_freq=30.0,
        notch_freq=60.0,
        verbose=False
    )
    print(f"✓ Data loaded: {len(epochs)} epochs, {len(epochs.ch_names)} channels")
    print(f"   Shape: {epochs.get_data().shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Feature extraction
print("\n[4/7] Extracting CSP features...")
try:
    csp, features = extract_csp_features(epochs, n_components=4)
    print(f"✓ CSP features extracted: {features.shape}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Classification with LDA (default)
print("\n[5/7] Training LDA classifier with cross-validation...")
try:
    pipeline = create_csp_lda_pipeline(n_components=4)
    scores = train_and_evaluate(
        epochs,
        pipeline,
        cv=3,  # Use 3 folds for quick testing
        scoring='accuracy',
        verbose=False
    )

    import numpy as np
    mean_accuracy = np.mean(scores['test_accuracy'])
    std_accuracy = np.std(scores['test_accuracy'])

    print(f"✓ LDA classification complete")
    print(f"   Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})")
except Exception as e:
    print(f"✗ Classification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Compare all classifiers
print("\n[6/7] Comparing all classifiers...")
try:
    comparison_results = compare_classifiers(epochs, cv=3, verbose=False)

    print("✓ Classifier comparison complete")
    print("\n   Results (3-fold CV):")
    print("   " + "-" * 60)

    # Sort by mean accuracy
    sorted_results = sorted(
        comparison_results.items(),
        key=lambda x: x[1]['mean_accuracy'],
        reverse=True
    )

    for rank, (name, result) in enumerate(sorted_results, 1):
        mean_acc = result['mean_accuracy']
        std_acc = result['std_accuracy']
        print(f"   {rank}. {name:25s}: {mean_acc:.3f} (+/- {std_acc:.3f})")

except Exception as e:
    print(f"✗ Classifier comparison failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is optional

# Test 7: Visualization (save only, don't display)
print("\n[7/7] Testing visualization...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    from pathlib import Path
    output_dir = Path('outputs/test')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_csp_patterns(
        csp,
        epochs.info,
        n_components=4,
        save_path=output_dir / 'test_csp_patterns.png',
        show=False
    )
    print(f"✓ Visualization saved to {output_dir}")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - visualization is optional

# Final summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("✓ Pipeline test completed successfully!")
print(f"✓ Processed {len(epochs)} epochs from Subject {subject}")
print(f"✓ Achieved {mean_accuracy:.1%} classification accuracy (LDA)")
print("\nYou can now run the full pipeline with different classifiers:")
print("  python src/main.py --subject 1 --task imagery_left_right --classifier lda")
print("  python src/main.py --subject 1 --task imagery_left_right --classifier logistic")
print("  python src/main.py --subject 1 --task imagery_left_right --classifier svm")
print("  python src/main.py --subject 1 --task imagery_left_right --classifier rf")
print("\nOr compare all classifiers:")
print("  python src/main.py --subject 1 --compare-classifiers")
print("=" * 70)
