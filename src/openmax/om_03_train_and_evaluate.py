
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pickle
import time
from om_01_model import OpenMaxClassifier
import warnings
warnings.filterwarnings('ignore')


def create_open_set_scenario(X, y, classes, known_classes_ratio=0.7, random_state=42):
    """
    Create open set scenario by selecting a subset of classes as known classes.
    """
    np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    num_known_classes = int(len(unique_classes) * known_classes_ratio)
    
    # Select known classes (ensure we have enough samples for each)
    class_counts = pd.Series(y).value_counts()
    # Sort by count and take top classes to ensure sufficient samples
    known_classes = class_counts.head(num_known_classes).index.values
    unknown_classes = np.setdiff1d(unique_classes, known_classes)
    
    # Split data
    known_mask = np.isin(y, known_classes)
    unknown_mask = np.isin(y, unknown_classes)
    
    X_known = X[known_mask]
    y_known = y[known_mask]
    X_unknown = X[unknown_mask]
    y_unknown = y[unknown_mask]
    
    # Remap known class labels to 0, 1, 2, ...
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(known_classes)}
    y_known_remapped = np.array([label_mapping[label] for label in y_known])
    
    return X_known, y_known_remapped, X_unknown, y_unknown, known_classes, unknown_classes, label_mapping


def evaluate_fold(classifier, X_test, y_test, known_classes):
    """
    Evaluate a single fold and return detailed metrics.
    """
    results = classifier.evaluate(X_test, y_test)
    
    # Calculate additional metrics
    predictions = results['predictions']
    
    # Known class accuracy (excluding unknown predictions)
    known_mask = y_test < len(known_classes)
    if np.sum(known_mask) > 0:
        known_predictions = predictions[known_mask]
        known_true = y_test[known_mask]
        known_accuracy = accuracy_score(known_true, known_predictions)
    else:
        known_accuracy = 0
    
    # Unknown detection metrics
    unknown_mask = y_test >= len(known_classes)
    if np.sum(unknown_mask) > 0:
        unknown_predictions = predictions[unknown_mask]
        unknown_detected = np.sum(unknown_predictions == len(known_classes))
        unknown_detection_rate = unknown_detected / np.sum(unknown_mask)
    else:
        unknown_detection_rate = 0
    
    # Overall metrics
    y_test_extended = y_test.copy()
    y_test_extended[unknown_mask] = len(known_classes)
    
    overall_accuracy = accuracy_score(y_test_extended, predictions)
    
    # Macro F1 score
    try:
        macro_f1 = f1_score(y_test_extended, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test_extended, predictions, average='weighted', zero_division=0)
    except:
        macro_f1 = 0
        weighted_f1 = 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'known_accuracy': known_accuracy,
        'unknown_detection_rate': unknown_detection_rate,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': predictions,
        'probabilities': results['probabilities'],
        'confusion_matrix': results['confusion_matrix']
    }


def cross_validation_evaluation(X_known, y_known, X_unknown, y_unknown, known_classes, best_params, cv_folds=5):
    """
    Perform cross-validation evaluation with the best parameters.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = []
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_known, y_known)):
        print(f"Evaluating fold {fold + 1}/{cv_folds}")
        
        # Split known data
        X_train, X_val = X_known[train_idx], X_known[val_idx]
        y_train, y_val = y_known[train_idx], y_known[val_idx]
        
        # Create balanced test set with validation data + unknown data
        # Use proportional sampling for unknown data
        unknown_sample_size = min(len(X_unknown), len(X_val) * 2)  # 2:1 ratio
        unknown_indices = np.random.choice(len(X_unknown), unknown_sample_size, replace=False)
        X_unknown_sample = X_unknown[unknown_indices]
        
        X_test = np.vstack([X_val, X_unknown_sample])
        y_test = np.hstack([y_val, np.full(len(X_unknown_sample), len(known_classes))])  # Unknown class label
        
        # Train OpenMax classifier
        classifier = OpenMaxClassifier(
            input_dim=X_train.shape[1],
            num_known_classes=len(known_classes),
            hidden_dims=best_params['hidden_dims'],
            alpha=best_params['alpha'],
            distance_type=best_params['distance_type'],
            tailsize=best_params['tailsize']
        )
        
        # Train base model
        training_history = classifier.train_base_model(
            X_train, y_train,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            lr=best_params['lr'],
            patience=best_params['patience']
        )
        
        # Fit OpenMax layer
        classifier.fit_openmax_layer(X_train, y_train)
        
        # Evaluate fold
        fold_results = evaluate_fold(classifier, X_test, y_test, known_classes)
        cv_results.append(fold_results)
        
        # Store predictions and labels for later analysis
        all_predictions.extend(fold_results['predictions'])
        all_true_labels.extend(y_test)
        all_probabilities.extend(fold_results['probabilities'])
        
        print(f"Fold {fold + 1} - Overall Accuracy: {fold_results['overall_accuracy']:.4f}, "
              f"Known Accuracy: {fold_results['known_accuracy']:.4f}, "
              f"Unknown Detection Rate: {fold_results['unknown_detection_rate']:.4f}")
    
    return cv_results, all_predictions, all_true_labels, all_probabilities


def final_model_training(X_known, y_known, X_unknown, y_unknown, known_classes, best_params):
    """
    Train the final model on all available data and evaluate on a held-out test set.
    """
    print("Training final model on all data...")
    
    # Split data into train and test
    from sklearn.model_selection import train_test_split
    
    # Split known data
    X_known_train, X_known_test, y_known_train, y_known_test = train_test_split(
        X_known, y_known, test_size=0.2, random_state=42, stratify=y_known
    )
    
    # Split unknown data
    X_unknown_train, X_unknown_test, y_unknown_train, y_unknown_test = train_test_split(
        X_unknown, y_unknown, test_size=0.2, random_state=42
    )
    
    # Create final test set
    X_test_final = np.vstack([X_known_test, X_unknown_test])
    y_test_final = np.hstack([y_known_test, np.full(len(X_unknown_test), len(known_classes))])
    
    # Train final classifier
    final_classifier = OpenMaxClassifier(
        input_dim=X_known_train.shape[1],
        num_known_classes=len(known_classes),
        hidden_dims=best_params['hidden_dims'],
        alpha=best_params['alpha'],
        distance_type=best_params['distance_type'],
        tailsize=best_params['tailsize']
    )
    
    # Train base model with validation
    training_history = final_classifier.train_base_model(
        X_known_train, y_known_train,
        X_val=X_known_test[:len(X_known_test)//2],  # Use part of test set for validation
        y_val=y_known_test[:len(y_known_test)//2],
        epochs=best_params['epochs'] * 2,  # Train longer for final model
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        patience=best_params['patience'] * 2
    )
    
    # Fit OpenMax layer
    final_classifier.fit_openmax_layer(X_known_train, y_known_train)
    
    # Final evaluation
    final_results = evaluate_fold(final_classifier, X_test_final, y_test_final, known_classes)
    
    return final_classifier, final_results, training_history


def save_results(cv_results, final_results, training_history, best_params, known_classes, unknown_classes):
    """
    Save all results and create summary statistics.
    """
    # Calculate CV statistics
    cv_stats = {
        'overall_accuracy': {
            'mean': np.mean([r['overall_accuracy'] for r in cv_results]),
            'std': np.std([r['overall_accuracy'] for r in cv_results])
        },
        'known_accuracy': {
            'mean': np.mean([r['known_accuracy'] for r in cv_results]),
            'std': np.std([r['known_accuracy'] for r in cv_results])
        },
        'unknown_detection_rate': {
            'mean': np.mean([r['unknown_detection_rate'] for r in cv_results]),
            'std': np.std([r['unknown_detection_rate'] for r in cv_results])
        },
        'macro_f1': {
            'mean': np.mean([r['macro_f1'] for r in cv_results]),
            'std': np.std([r['macro_f1'] for r in cv_results])
        },
        'weighted_f1': {
            'mean': np.mean([r['weighted_f1'] for r in cv_results]),
            'std': np.std([r['weighted_f1'] for r in cv_results])
        }
    }
    
    # Save detailed results
    results_dict = {
        'cv_results': cv_results,
        'cv_statistics': cv_stats,
        'final_results': final_results,
        'training_history': training_history,
        'best_params': best_params,
        'known_classes': known_classes,
        'unknown_classes': unknown_classes
    }
    
    with open('results/training_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    # Create summary DataFrame
    summary_data = {
        'Metric': ['Overall Accuracy', 'Known Class Accuracy', 'Unknown Detection Rate', 'Macro F1', 'Weighted F1'],
        'CV_Mean': [cv_stats[key]['mean'] for key in ['overall_accuracy', 'known_accuracy', 'unknown_detection_rate', 'macro_f1', 'weighted_f1']],
        'CV_Std': [cv_stats[key]['std'] for key in ['overall_accuracy', 'known_accuracy', 'unknown_detection_rate', 'macro_f1', 'weighted_f1']],
        'Final_Model': [
            final_results['overall_accuracy'],
            final_results['known_accuracy'],
            final_results['unknown_detection_rate'],
            final_results['macro_f1'],
            final_results['weighted_f1']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/training_evaluation_summary.csv', index=False)
    
    return cv_stats, summary_df

X = np.load("data/processed/openmax/X.npy")
y = np.load("data/processed/openmax/y.npy")
classes = np.load("data/processed/openmax/classes.npy", allow_pickle=True)

# Load best parameters
try:
    with open('results/best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)
    print(f"Loaded best parameters: {best_params}")
except FileNotFoundError:
    print("Best parameters not found. Using default parameters.")
    best_params = {
        'hidden_dims': [128, 64, 32],
        'alpha': 10,
        'distance_type': 'euclidean',
        'tailsize': 20,
        'epochs': 100,
        'batch_size': 256,
        'lr': 0.001,
        'patience': 15
    }



# Create open set scenario
print("Creating open set scenario...")
X_known, y_known, X_unknown, y_unknown, known_classes, unknown_classes, label_mapping = create_open_set_scenario(
    X, y, classes, known_classes_ratio=0.7, random_state=42
)

print(f"Known classes: {len(known_classes)} ({known_classes})")
print(f"Unknown classes: {len(unknown_classes)}")
print(f"Known data shape: {X_known.shape}")
print(f"Unknown data shape: {X_unknown.shape}")

# Cross-validation evaluation
print("\nPerforming cross-validation evaluation...")
start_time = time.time()
cv_results, all_predictions, all_true_labels, all_probabilities = cross_validation_evaluation(
    X_known, y_known, X_unknown, y_unknown, known_classes, best_params
)
cv_time = time.time() - start_time
print(f"Cross-validation completed in {cv_time:.2f} seconds")

# Final model training
print("\nTraining final model...")
start_time = time.time()
final_classifier, final_results, training_history = final_model_training(
    X_known, y_known, X_unknown, y_unknown, known_classes, best_params
)
final_training_time = time.time() - start_time
print(f"Final model training completed in {final_training_time:.2f} seconds")

# Save results
print("\nSaving results...")
cv_stats, summary_df = save_results(cv_results, final_results, training_history, 
                                    best_params, known_classes, unknown_classes)