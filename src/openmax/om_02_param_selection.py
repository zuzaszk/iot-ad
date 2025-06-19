import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import itertools
import time
import pickle
from om_01_model import OpenMaxClassifier
import warnings
warnings.filterwarnings('ignore')


def create_open_set_scenario(X, y, classes, known_classes_ratio=0.7, random_state=42):
    """
    Create open set scenario by selecting a subset of classes as known classes.
    
    Args:
        X: Feature matrix
        y: Labels
        classes: Class names
        known_classes_ratio: Ratio of classes to be considered as known
        random_state: Random seed
    
    Returns:
        X_known, y_known, X_unknown, y_unknown, known_classes, unknown_classes
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


def evaluate_openmax_cv(X_known, y_known, X_unknown, y_unknown, params, cv_folds=5, random_state=42):
    """
    Evaluate OpenMax using cross-validation on known classes and test on unknown classes.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_known, y_known)):
        print(f"  Fold {fold + 1}/{cv_folds}")
        
        # Split known data
        X_train, X_val = X_known[train_idx], X_known[val_idx]
        y_train, y_val = y_known[train_idx], y_known[val_idx]
        
        # Create test set with validation known data + some unknown data
        # Use smaller portion of unknown data for faster evaluation
        unknown_sample_size = min(len(X_unknown), len(X_val))
        unknown_indices = np.random.choice(len(X_unknown), unknown_sample_size, replace=False)
        X_unknown_sample = X_unknown[unknown_indices]
        
        X_test = np.vstack([X_val, X_unknown_sample])
        y_test = np.hstack([y_val, np.full(len(X_unknown_sample), len(np.unique(y_known)))])  # Unknown class label
        
        # Train OpenMax
        classifier = OpenMaxClassifier(
            input_dim=X_train.shape[1],
            num_known_classes=len(np.unique(y_known)),
            hidden_dims=params['hidden_dims'],
            alpha=params['alpha'],
            distance_type=params['distance_type'],
            tailsize=params['tailsize']
        )
        
        # Train base model
        classifier.train_base_model(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['lr'],
            patience=params['patience']
        )
        
        # Fit OpenMax layer
        classifier.fit_openmax_layer(X_train, y_train)
        
        # Evaluate
        results = classifier.evaluate(X_test, y_test)
        cv_scores.append(results['accuracy'])
    
    return np.mean(cv_scores), np.std(cv_scores)


def parameter_selection():
    """
    Perform parameter selection for OpenMax using cross-validation.
    """
    print("Loading data...")
    X = np.load("data/processed/openmax/X.npy")
    y = np.load("data/processed/openmax/y.npy")
    classes = np.load("data/processed/openmax/classes.npy", allow_pickle=True)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create open set scenario
    print("Creating open set scenario...")
    X_known, y_known, X_unknown, y_unknown, known_classes, unknown_classes, label_mapping = create_open_set_scenario(
        X, y, classes, known_classes_ratio=0.7, random_state=42
    )
    
    print(f"Known classes: {len(known_classes)}")
    print(f"Unknown classes: {len(unknown_classes)}")
    print(f"Known data: {X_known.shape}")
    print(f"Unknown data: {X_unknown.shape}")
    
    # Define parameter grid
    param_grid = {
        'hidden_dims': [
            [64, 32],
            # [128, 64, 32],
            [256, 128, 64]
        ],
        'alpha': [10],
        'distance_type': ['euclidean', 'cosine'],
        'tailsize': [10, 30],
        'epochs': [50],  # Reduced for faster parameter search
        'batch_size': [256],
        'lr': [0.001],
        'patience': [10]
    }
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # Parameter search
    results = []
    best_score = 0
    best_params = None
    
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        print(f"\nTesting combination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        start_time = time.time()
        
        try:
            mean_score, std_score = evaluate_openmax_cv(X_known, y_known, X_unknown, y_unknown, params)
            
            results.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'time': time.time() - start_time
            })
            
            print(f"Mean accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
            print(f"Time: {time.time() - start_time:.2f} seconds")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                print(f"New best score: {best_score:.4f}")
                
        except Exception as e:
            print(f"Error with parameters: {e}")
            results.append({
                'params': params.copy(),
                'mean_score': 0,
                'std_score': 0,
                'time': time.time() - start_time,
                'error': str(e)
            })
    
    # Save results
    print("\nSaving parameter selection results...")
    
    # Save detailed results
    with open('results/parameter_selection_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'best_params': best_params,
            'best_score': best_score,
            'known_classes': known_classes,
            'unknown_classes': unknown_classes,
            'label_mapping': label_mapping
        }, f)
    
    # Save best parameters separately
    with open('results/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    # Create results summary
    results_df = pd.DataFrame([
        {
            **r['params'],
            'mean_score': r['mean_score'],
            'std_score': r['std_score'],
            'time': r['time']
        } for r in results if 'error' not in r
    ])
    
    results_df.to_csv('results/parameter_selection_summary.csv', index=False)
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    # Display top 10 results
    print("\nTop 10 parameter combinations:")
    top_results = results_df.nlargest(10, 'mean_score')
    print(top_results[['hidden_dims', 'alpha', 'distance_type', 'tailsize', 'lr', 'mean_score', 'std_score']])
    
    return best_params, results


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    print("Starting OpenMax parameter selection...")
    best_params, results = parameter_selection()
    print("Parameter selection completed!")