import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import weibull_min
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class BaseModel(nn.Module):
    def __init__(self, input_dim, num_known_classes, hidden_dims=[64, 32, 16]):
        super(BaseModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_known_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

class OpenMaxLayer:
    """
    OpenMax layer implementation for open set recognition.
    """
    def __init__(self, num_known_classes, alpha=10, distance_type='euclidean'):
        self.num_known_classes = num_known_classes
        self.alpha = alpha  # Number of top classes to revise
        self.distance_type = distance_type
        self.mean_activation_vectors = {}
        self.weibull_models = {}
        
    def fit_weibull(self, features, labels, tailsize=20):
        """
        Fit Weibull distribution for each class based on distances from mean activation vector.
        
        Args:
            features: Feature vectors from the penultimate layer
            labels: True class labels
            tailsize: Number of largest distances to use for Weibull fitting
        """
        # Calculate mean activation vectors for each class
        for class_idx in range(self.num_known_classes):
            class_features = features[labels == class_idx]
            if len(class_features) > 0:
                self.mean_activation_vectors[class_idx] = np.mean(class_features, axis=0)
        
        # Fit Weibull distribution for each class
        for class_idx in range(self.num_known_classes):
            if class_idx in self.mean_activation_vectors:
                class_features = features[labels == class_idx]
                
                # Calculate distances from mean activation vector
                if self.distance_type == 'euclidean':
                    distances = [euclidean(feat, self.mean_activation_vectors[class_idx]) 
                               for feat in class_features]
                elif self.distance_type == 'cosine':
                    distances = [cosine(feat, self.mean_activation_vectors[class_idx]) 
                               for feat in class_features]
                else:
                    raise ValueError("Distance type must be 'euclidean' or 'cosine'")
                
                # Use largest distances for Weibull fitting (tailsize)
                distances = np.array(distances)
                if len(distances) >= tailsize:
                    tail_distances = np.sort(distances)[-tailsize:]
                else:
                    tail_distances = distances
                
                # Fit Weibull distribution
                if len(tail_distances) > 1 and np.std(tail_distances) > 0:
                    try:
                        shape, loc, scale = weibull_min.fit(tail_distances, floc=0)
                        self.weibull_models[class_idx] = {'shape': shape, 'scale': scale}
                    except:
                        # Fallback parameters if fitting fails
                        self.weibull_models[class_idx] = {'shape': 1.0, 'scale': np.mean(tail_distances)}
                else:
                    # Fallback for insufficient data
                    self.weibull_models[class_idx] = {'shape': 1.0, 'scale': np.mean(tail_distances) if len(tail_distances) > 0 else 1.0}
    
    def compute_openmax_prob(self, logits, features):
        """
        Compute OpenMax probabilities including the unknown class.
        
        Args:
            logits: Output logits from the classifier
            features: Feature vectors from the penultimate layer
            
        Returns:
            OpenMax probabilities (including unknown class probability)
        """
        batch_size = logits.shape[0]
        openmax_probs = []
        
        for i in range(batch_size):
            logit = logits[i]
            feature = features[i]
            
            # Get top alpha classes
            top_classes = np.argsort(logit)[-self.alpha:]
            
            # Compute revised logits
            revised_logits = logit.copy()
            unknowness_score = 0
            
            for class_idx in top_classes:
                if class_idx in self.mean_activation_vectors and class_idx in self.weibull_models:
                    # Calculate distance to mean activation vector
                    if self.distance_type == 'euclidean':
                        distance = euclidean(feature, self.mean_activation_vectors[class_idx])
                    else:
                        distance = cosine(feature, self.mean_activation_vectors[class_idx])
                    
                    # Calculate Weibull CDF (probability of being this far or farther)
                    weibull_params = self.weibull_models[class_idx]
                    prob_incl = weibull_min.cdf(distance, 
                                              weibull_params['shape'], 
                                              scale=weibull_params['scale'])
                    
                    # Revise logit
                    w_score = 1 - prob_incl
                    revised_logits[class_idx] = logit[class_idx] * w_score
                    unknowness_score += logit[class_idx] * (1 - w_score)
            
            # Convert to probabilities using softmax
            exp_logits = np.exp(revised_logits - np.max(revised_logits))
            exp_unknowness = np.exp(unknowness_score - np.max(revised_logits)) if unknowness_score > 0 else 0
            
            total = np.sum(exp_logits) + exp_unknowness
            
            # OpenMax probabilities (known classes + unknown class)
            openmax_prob = np.zeros(self.num_known_classes + 1)
            openmax_prob[:-1] = exp_logits / total
            openmax_prob[-1] = exp_unknowness / total  # Unknown class probability
            
            openmax_probs.append(openmax_prob)
        
        return np.array(openmax_probs)


class OpenMaxClassifier:
    """
    Complete OpenMax classifier combining base model and OpenMax layer.
    """
    def __init__(self, input_dim, num_known_classes, hidden_dims=[128, 64, 32], 
                 alpha=10, distance_type='euclidean', tailsize=20):
        self.input_dim = input_dim
        self.num_known_classes = num_known_classes
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.distance_type = distance_type
        self.tailsize = tailsize
        
        # Initialize models
        self.base_model = BaseModel(input_dim, num_known_classes, hidden_dims)
        self.openmax_layer = OpenMaxLayer(num_known_classes, alpha, distance_type)
        
        # Training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        
    def train_base_model(self, X_train, y_train, X_val=None, y_val=None, 
                        epochs=100, batch_size=256, lr=0.001, patience=10):
        """
        Train the base neural network model.
        """
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.base_model.train()
        training_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits, _ = self.base_model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.base_model.eval()
                with torch.no_grad():
                    val_logits, _ = self.base_model(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()
                    val_pred = torch.argmax(val_logits, dim=1)
                    val_acc = (val_pred == y_val_tensor).float().mean().item()
                
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.base_model.state_dict(), 'best_base_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    # Load best model
                    self.base_model.load_state_dict(torch.load('best_base_model.pth'))
                    break
                
                self.base_model.train()
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}')
        
        return training_history
    
    def fit_openmax_layer(self, X_train, y_train):
        """
        Fit the OpenMax layer using training data.
        """
        self.base_model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            _, features = self.base_model(X_train_tensor)
            features = features.cpu().numpy()
        
        self.openmax_layer.fit_weibull(features, y_train, self.tailsize)
    
    def predict(self, X, return_features=False):
        """
        Make predictions using OpenMax.
        
        Returns:
            predictions: Class predictions (num_known_classes for unknown)
            probabilities: OpenMax probabilities
            features: Feature vectors (if return_features=True)
        """
        self.base_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits, features = self.base_model(X_tensor)
            logits = logits.cpu().numpy()
            features = features.cpu().numpy()
        
        # Compute OpenMax probabilities
        openmax_probs = self.openmax_layer.compute_openmax_prob(logits, features)
        
        # Get predictions (unknown class has index num_known_classes)
        predictions = np.argmax(openmax_probs, axis=1)
        
        if return_features:
            return predictions, openmax_probs, features
        else:
            return predictions, openmax_probs
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate the OpenMax classifier.
        """
        predictions, probabilities = self.predict(X_test)
        
        # Convert unknown predictions to a special label
        y_test_extended = y_test.copy()
        
        # For evaluation, we need to handle the case where test set might contain unknown classes
        # If y_test contains classes >= num_known_classes, treat them as unknown
        unknown_mask = y_test >= self.num_known_classes
        y_test_extended[unknown_mask] = self.num_known_classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_extended, predictions)
        
        # Create extended class names
        if class_names is not None:
            extended_class_names = list(class_names[:self.num_known_classes]) + ['UNKNOWN']
        else:
            extended_class_names = [f'Class_{i}' for i in range(self.num_known_classes)] + ['UNKNOWN']
        
        report = classification_report(y_test_extended, predictions, 
                                     target_names=extended_class_names, 
                                     zero_division=0)
        
        cm = confusion_matrix(y_test_extended, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }