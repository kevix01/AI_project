from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from typing import Optional, List, Tuple, Dict, Union, Any
import numpy as np
from sklearn.model_selection import KFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LogHybridLoss(nn.Module):
    def __init__(self, alpha=0.6, epsilon=1e-6):
        super(LogHybridLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        # MSE classico
        mse = torch.mean((y_pred - y_true) ** 2)

        # Log-relative error
        log_y_pred = torch.log1p(torch.abs(y_pred) + self.epsilon)
        log_y_true = torch.log1p(torch.abs(y_true) + self.epsilon)
        log_rel_mse = torch.mean((log_y_pred - log_y_true) ** 2)

        return self.alpha * mse + (1 - self.alpha) * log_rel_mse

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size1: int, hidden_size2: int) -> None:
        super(SimpleMLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, 1)
        self.reinitialize_weights()
        logger.info(f"Initialized SimpleMLP with hidden_size1={hidden_size1}, hidden_size2={hidden_size2}")

    def forward(self, x: Union[float, int, List[float], torch.Tensor]) -> Union[float, torch.Tensor]:
        if isinstance(x, (int, float)):
            x = torch.tensor([[x]], dtype=torch.float32)
        elif isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)

        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x.item() if x.numel() == 1 else x

    def reinitialize_weights(self) -> None:
        for name, layer in [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('output', self.output)]:
            #nn.init.normal_(layer.weight, mean=0, std=1)
            nn.init.uniform_(layer.weight, -1.0, 1.0)
            nn.init.uniform_(layer.bias, -1.0, 1.0)
            logger.debug(f"Reinitialized weights for layer {name}")

class TrainableMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int) -> None:
        super(TrainableMLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, 1)
        self.reinitialize_weights()
        logger.info(f"Initialized TrainableMLP with input_size={input_size}, hidden_size1={hidden_size1}")

    def forward(self, x: Union[float, int, List[float], torch.Tensor]) -> torch.Tensor | List[float] | float:
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32).view(1, -1)
        elif isinstance(x, (int, float)):
            x = torch.tensor([[x]], dtype=torch.float32)
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.output(x)
        return x

    def reinitialize_weights(self, seed: Optional[int] = None) -> None:
        """Reinitialize weights with optional seed (locally scoped)"""
        if seed is not None:
            # Save the current RNG state
            original_state = torch.random.get_rng_state()
            
            # Set the new seed (local to this block)
            torch.manual_seed(seed)
            
            try:
                for name, layer in [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('output', self.output)]:
                    nn.init.uniform_(layer.weight, -1.0, 1.0)
                    nn.init.uniform_(layer.bias, -1.0, 1.0)
                    logger.debug(f"Reinitialized weights for layer {name} (seed={seed})")
            finally:
                # Restore the original RNG state
                torch.random.set_rng_state(original_state)
        else:
            # No seed provided - just do normal initialization
            for name, layer in [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('output', self.output)]:
                nn.init.uniform_(layer.weight, -1.0, 1.0)
                nn.init.uniform_(layer.bias, -1.0, 1.0)

    def train_model(
        self, 
        inputs: List[List[float]], 
        targets: List[float], 
        val_inputs: Optional[List[List[float]]] = None,
        val_targets: Optional[List[float]] = None,
        lr: float = 0.01, 
        epochs: int = 100, 
        batch_size: int = 1,
        verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model with optional validation at each epoch.
        Returns training and validation loss histories.
        """
        criterion = LogHybridLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        dataset_size = inputs_tensor.size(0)
        
        # Prepare validation data if provided
        val_loss_history: List[float] = []
        if val_inputs is not None and val_targets is not None:
            val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1)
            if verbose:
                logger.info("Validation data provided - will validate at each epoch")
        
        train_loss_history: List[float] = []
        if verbose:
            logger.info(f"Starting training for {epochs} epochs with batch_size={batch_size}, lr={lr}")
        
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0

            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_inputs = inputs_tensor[indices]
                batch_targets = targets_tensor[indices]

                optimizer.zero_grad()
                outputs = self.forward(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / (dataset_size / batch_size)
            train_loss_history.append(avg_epoch_loss)
            
            # Validation phase
            if val_inputs is not None and val_targets is not None:
                val_loss, est_integral = self.validate_model(val_inputs, val_targets)
                val_loss_history.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), "weights_TrainableMLP.pth")
                    if verbose:
                        logger.info(f"New best validation loss: {best_val_loss:.6f} at epoch {epoch+1}")
                if verbose and (epoch+1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {avg_epoch_loss:.6f} - "
                        f"Val Loss: {val_loss:.6f}"
                    )
            elif verbose and (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.6f}")

        if verbose:
            logger.info(f"Training completed. Final Train Loss: {train_loss_history[-1]:.6f}")
            if val_loss_history:
                logger.info(f"Final Validation Loss: {val_loss_history[-1]:.6f}")
        
        return train_loss_history, val_loss_history

    def validate_model(self, val_inputs: List[List[float]], val_targets: List[float]) -> float:
        self.eval()
        with torch.no_grad():
            inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1)
            outputs = self.forward(inputs_tensor)
            loss = nn.MSELoss()(outputs, targets_tensor)
        return loss.item(), outputs

def sample_uniform(n: int) -> List[float]:
    """Generate n uniformly distributed samples between 0 and 1, sorted ascending."""
    samples = torch.empty(n).uniform_(0, 1).tolist()
    return sorted(samples)

def trapezoidal_integral(x: List[float], y: List[float]) -> float:
    """
    Compute the integral using the trapezoidal rule.
    
    Args:
        x: List of x-coordinates (must be sorted ascending)
        y: List of y-coordinates corresponding to x
        
    Returns:
        Approximate integral value
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two points are needed for integration.")

    integral = 0.0
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        area = h * (y[i] + y[i - 1]) / 2
        integral += area
    return integral

def plot_function(x: List[float], y: List[float], title: str = "Function Plot", 
                 xlabel: str = "x", ylabel: str = "f(x)") -> None:
    """Plot a function given x and y coordinates."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_losses(train_losses: List[float], val_losses: List[float], title: str = "Training and Validation Loss") -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_training_data(L: List[float], mlp: SimpleMLP, dataset_size: int) -> Tuple[List[List[float]], List[float]]:
    """Generate training data using SimpleMLP."""
    logger.info(f"Generating {dataset_size} training samples...")
    train_data: List[List[float]] = []
    train_targets: List[float] = []
    
    for i in range(dataset_size):
        L1: List[float] = []
        for el in L:
            output = mlp.forward(el)
            L1.append(output.item() if isinstance(output, torch.Tensor) else output)
        mlp.reinitialize_weights()
        train_data.append(L1)
        target = trapezoidal_integral(L, L1)
        train_targets.append(target)
        
        if (i+1) % 40 == 0:
            logger.info(f"Generated {i+1}/{dataset_size} training samples")
    
    logger.info("Training data generation completed")
    return train_data, train_targets

def kfold_cross_validation(
    model: TrainableMLP,
    inputs: List[List[float]], 
    targets: List[float],
    k: int = 5,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 1,
    random_state: int = 42
) -> Dict[str, Dict[str, Union[List[List[float]], List[float], Any]]]:
    """
    Perform k-fold cross validation on the given model and data.
    Returns dictionary with average metrics across all folds.
    """
    logger.info(f"Starting {k}-fold cross validation with {epochs} epochs per fold")
    
    # Convert to numpy arrays for KFold
    inputs_np = np.array(inputs)
    targets_np = np.array(targets)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results: Dict[str, List] = {
        'train_loss': [],    # This will contain List[List[float]]
        'val_loss': [],      # This will contain List[List[float]]
        'final_train_loss': [],  # This will contain List[float]
        'final_val_loss': []     # This will contain List[float]
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(inputs_np), 1):
        logger.info(f"=== Starting Fold {fold}/{k} ===")
        
        # Split data
        train_inputs = inputs_np[train_idx].tolist()
        train_targets = targets_np[train_idx].tolist()
        val_inputs = inputs_np[val_idx].tolist()
        val_targets = targets_np[val_idx].tolist()

        # Different seed per fold to ensure varied but reproducible init
        fold_seed = random_state + fold
        
        # Reinitialize model weights for each fold
        model.reinitialize_weights(seed=fold_seed)
        
        # Train model
        train_loss, val_loss = model.train_model(
            train_inputs,
            train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False  # Less verbose for k-fold
        )
        
        # Store results
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1] if val_loss else model.validate_model(val_inputs, val_targets)
        
        fold_results['train_loss'].append(train_loss)
        fold_results['val_loss'].append(val_loss)
        fold_results['final_train_loss'].append(final_train_loss)
        fold_results['final_val_loss'].append(final_val_loss)
        
        logger.info(
            f"Fold {fold} completed - "
            f"Final Train Loss: {final_train_loss:.6f} - "
            f"Final Val Loss: {final_val_loss:.6f}"
        )
    
    # Calculate average metrics
    avg_metrics: Dict[str, float] = {
        'avg_final_train_loss': float(np.mean(fold_results['final_train_loss'])),
        'avg_final_val_loss': float(np.mean(fold_results['final_val_loss'])),
        'std_final_train_loss': float(np.std(fold_results['final_train_loss'])),
        'std_final_val_loss': float(np.std(fold_results['final_val_loss']))
    }
    
    logger.info("=== Cross Validation Results ===")
    logger.info(f"Average Final Training Loss: {avg_metrics['avg_final_train_loss']:.6f} ± {avg_metrics['std_final_train_loss']:.6f}")
    logger.info(f"Average Final Validation Loss: {avg_metrics['avg_final_val_loss']:.6f} ± {avg_metrics['std_final_val_loss']:.6f}")
    
    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics
    }

def plot_all_fold_performances(fold_results: Dict[str, Union[List[List[float]], List[float]]]) -> None:
    """Plot training and validation curves for all folds with multiple visualizations."""
    num_folds = len(fold_results['train_loss'])
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 6))
    
    # First subplot: Individual folds
    plt.subplot(1, 2, 1)
    for i, (train_loss, val_loss) in enumerate(zip(fold_results['train_loss'], fold_results['val_loss']), 1):
        plt.plot(train_loss, '-', alpha=0.7, label=f'Train Fold {i}')
        plt.plot(val_loss, '--', alpha=0.7, label=f'Val Fold {i}')
    
    plt.title(f"Individual Fold Performances (n={num_folds})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Second subplot: Average curves
    plt.subplot(1, 2, 2)
    # Plot individual fold curves with low opacity
    for train_loss, val_loss in zip(fold_results['train_loss'], fold_results['val_loss']):
        plt.plot(train_loss, 'b-', alpha=0.1)
        plt.plot(val_loss, 'r-', alpha=0.1)
    
    # Calculate and plot average curves
    avg_train_loss = np.mean(fold_results['train_loss'], axis=0)
    avg_val_loss = np.mean(fold_results['val_loss'], axis=0)
    
    plt.plot(avg_train_loss, 'b-', linewidth=2, label='Avg Train Loss')
    plt.plot(avg_val_loss, 'r-', linewidth=2, label='Avg Val Loss')
    
    plt.title(f"Average Performance Across Folds (n={num_folds})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_individual_folds(fold_results: Dict[str, Union[List[List[float]], List[float]]]) -> None:
    """Plot each fold's performance in a separate subplot."""
    num_folds = len(fold_results['train_loss'])
    cols = 2  # Number of columns in subplot grid
    rows = (num_folds + 1) // cols  # Calculate needed rows
    
    plt.figure(figsize=(16, 6 * rows))
    
    for i, (train_loss, val_loss) in enumerate(zip(fold_results['train_loss'], fold_results['val_loss']), 1):
        plt.subplot(rows, cols, i)
        plt.plot(train_loss, 'b-', label='Train Loss')
        plt.plot(val_loss, 'r-', label='Validation Loss')
        plt.title(f"Fold {i} Performance")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    # Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    # Generate input samples
    L = sample_uniform(500)
    logger.info(f"Generated {len(L)} uniform samples between 0 and 1")
    
    # Initialize models
    mlp = SimpleMLP(hidden_size1=48, hidden_size2=24)
    mlp2 = TrainableMLP(input_size=len(L), hidden_size1=16, hidden_size2=8)
    
    # Generate training data
    dataset_size = 600
    train_data, train_targets = generate_training_data(L, mlp, dataset_size)
    
    # Perform k-fold cross validation
    cv_results = kfold_cross_validation(
        mlp2,
        train_data,
        train_targets,
        k=4,
        epochs=200,
        batch_size=20,
        lr=0.005
    )
    
    # Plot all fold performances - now shows both individual and average
    plot_all_fold_performances(cv_results['fold_results'])
    
    # Additional visualization: Each fold in its own subplot
    plot_individual_folds(cv_results['fold_results'])
    
    #Generation of validation data
    logger.info("Generating validation data...")
    val_inputs = []
    val_targets = []
    n_val = 50
    for t in range(n_val):
        L1: List[float] = []
        mlp.reinitialize_weights() #changing the function computed by the MLP1
        for el in L:
            output = mlp.forward(el)
            L1.append(output.item() if isinstance(output, torch.Tensor) else output)
        target = trapezoidal_integral(L, L1)
        val_inputs.append(L1)
        val_targets.append(target)
    
    # Train final model on all data
    logger.info("Training final model on full dataset...")
    mlp2.reinitialize_weights()
    train_loss, _ = mlp2.train_model(
        train_data,
        train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        epochs=500,
        batch_size=10,
        lr=0.005,
        verbose=True
    )

    #Loading the best found model for TrainableMLP
    mlp2.load_state_dict(torch.load("weights_TrainableMLP.pth", weights_only=True))  

    # Final Tests --> building the 22 test data
    # Evaluation of the model on the 22 test data
    # Plot results
    n_test = 22
    for t in range(n_test):
        L1: List[float] = []
        mlp.reinitialize_weights() #changing the function computed by the MLP1
        for el in L:
            output = mlp.forward(el)
            L1.append(output.item() if isinstance(output, torch.Tensor) else output)
        target = trapezoidal_integral(L, L1)
        final_loss, estimated_integral = mlp2.validate_model([L1], [target])
        logger.info(f"Test {t+1}/{n_test} - Numeric integral: {target} - Estimated integral: {estimated_integral.item()} - Loss: {final_loss:.6f}")
        if t>16:
            plot_function(L, L1, title="MLP1 Output Function")


if __name__ == "__main__":
    main()