import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
from typing import List, Tuple, Dict, Any

# ── CONFIGURATION (REVISED) ───────────────────────────────────────────────────
# Central place to modify hyperparameters
CONFIG = {
    "data_path": "loan.csv",        # [cite: 1] referenced via path
    "target_col": "CreditCard",         # [cite: 1]
    "seed": 42,
    "n_teachers": 5,                # Increased back to 5 for more robust aggregation (adjust if needed)
    "teacher_lr": 1e-5,             # Standard teacher LR
    "teacher_epochs": 50,           # Increased teacher epochs for better learning
    "t_hidden_dim": 128,            # Slightly increased teacher capacity
    "lap_scale": 0.001,             # *** VERY IMPORTANT: Start very low. STRONGLY recommend testing lap_scale = 0.0 first! ***
    "noise_dim": 128,               # Reduced noise dimension slightly
    "g_hidden_dim": 256,            # Kept G/S capacity moderate
    "s_hidden_dim": 256,            # Kept G/S capacity moderate
    "pate_gan_epochs": 300,         # Increased PATE-GAN training epochs
    "batch_size": 128,
    "g_lr": 3e-5,                   # Adjusted G LR (Try 1e-4, 5e-5, 1e-5 as alternatives)
    "s_lr": 1e-5,                   # Adjusted S LR (Try making it equal to g_lr, or 1/10th of g_lr)
    "log_every": 20,                # Log every 20 epochs
    "eval_teachers": True,          # Keep evaluating teachers
}

# ── DATA PARTITIONING ──────────────────────────────────────────────────────────
def partition_data(X: np.ndarray, y: np.ndarray, n_teachers: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split dataset into n_teachers disjoint partitions."""
    if not isinstance(X, np.ndarray): X = X.values
    if not isinstance(y, np.ndarray): y = y.values

    data_splits = []
    N = len(X)
    if N < n_teachers:
        raise ValueError("Number of data points must be >= number of teachers.")
    split_size = N // n_teachers
    indices = np.random.permutation(N)

    print(f"Total data: {N}, Teachers: {n_teachers}, Approx partition size: {split_size}")
    for i in range(n_teachers):
        start = i * split_size
        # Assign remaining data points to the last teacher
        end = (i + 1) * split_size if i < n_teachers - 1 else N
        idx = indices[start:end]
        if len(idx) == 0 and N >= n_teachers: # Avoid empty splits if possible
             print(f"Warning: Teacher {i} received 0 data points. Adjust n_teachers or data size.")
             continue # Skip empty split if absolutely necessary, though better to ensure data>teachers
        data_splits.append((X[idx], y[idx]))
    print(f"Created {len(data_splits)} non-empty data splits.")
    return data_splits

def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    """Create DataLoader for numpy arrays."""
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
         X_t = X.float() # Ensure correct type
    if not isinstance(y, torch.Tensor):
        y_t = torch.tensor(y, dtype=torch.long)
    else:
         y_t = y.long() # Ensure correct type

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ── NOISY AGGREGATION ───────────────────────────────────────────────────────────
def noisy_aggregate_multiclass(teacher_ensemble: List[nn.Module],
                               x_batch: torch.Tensor,
                               num_classes: int,
                               lap_scale: float = 0.01,
                               device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Aggregates teacher votes with Laplace noise for PATE."""
    x_batch = x_batch.to(device)
    all_votes = []
    for teacher in teacher_ensemble:
        teacher.eval()
        with torch.no_grad():
            # Get predicted class index directly
            preds = torch.argmax(teacher(x_batch), dim=1)
            all_votes.append(preds.cpu().numpy())

    # Transpose votes: rows are samples, columns are teachers
    all_votes = np.array(all_votes).T # Shape: (batch_size, n_teachers)
    agg_labels = []

    for votes_i in all_votes: # Iterate over samples in the batch
        # Count votes for each class
        counts = np.bincount(votes_i, minlength=num_classes).astype(np.float32)
        if lap_scale > 0:
             # Add Laplace noise to the counts
            noise = np.random.laplace(0, lap_scale, size=num_classes)
            noisy_counts = counts + noise
        else:
            noisy_counts = counts # No noise if lap_scale is 0 or less

        # Find the class with the maximum noisy count
        agg_labels.append(np.argmax(noisy_counts))

    return torch.tensor(agg_labels, dtype=torch.long, device=device)

# ── MODEL CLASSES ───────────────────────────────────────────────────────────────
class ConditionalGenerator(nn.Module):
    """Conditional Generator Network for PATE-GAN."""
    def __init__(self, noise_dim: int, data_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        input_dim = noise_dim + num_classes # Combine noise and one-hot label

        # Simpler architecture potentially better with moderate hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh() # Output matches MinMaxScaler range (-1, 1)
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        # Concatenate noise and labels
        gen_input = torch.cat([z, one_hot_labels], dim=1)
        return self.net(gen_input)

class Teacher(nn.Module):
    """Teacher Classifier Network."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        # Simpler architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # Sometimes BatchNorm hurts simple models
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), # Adding another layer
            # nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes) # Output layer for classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_teacher(self, loader: DataLoader, lr: float = 1e-4, epochs: int = 15, device: torch.device = torch.device('cpu')):
        """Trains a single teacher model."""
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        print(f"  Training teacher for {epochs} epochs...")
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = self(xb)
                loss = loss_fn(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            # print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}") # Uncomment for verbose teacher train loss
        print("  Teacher training finished.")

    def evaluate(self, loader: DataLoader, device: torch.device = torch.device('cpu')) -> float:
        """Evaluates the teacher's accuracy on the provided data loader."""
        self.to(device)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = self(xb)
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        if total == 0: return 0.0 # Avoid division by zero
        accuracy = 100 * correct / total
        return accuracy


class Student(nn.Module):
    """Student Classifier Network (Discriminator in PATE-GAN)."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        # num_classes here includes the K real classes + 1 fake class
        # Simpler architecture matching Generator
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # Often removed in discriminator
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ── PATE-GAN TRAINING LOOP ──────────────────────────────────────────────────────
def train_pate_gan(generator: ConditionalGenerator,
                   student: Student,
                   teachers: List[Teacher],
                   real_loader: DataLoader,
                   config: Dict[str, Any],
                   device: torch.device = torch.device('cpu')) -> Tuple[ConditionalGenerator, Student]:
    """Trains the PATE-GAN Generator and Student."""

    # Extract relevant config parameters
    K = config['num_classes']
    noise_dim = config['noise_dim']
    epochs = config['pate_gan_epochs']
    lap_scale = config['lap_scale']
    g_lr = config['g_lr']
    s_lr = config['s_lr']
    log_every = config['log_every']

    generator.to(device)
    student.to(device)

    # Separate optimizers for Generator and Student
    # Using RMSprop or lower learning rate Adam might help stabilize GANs
    # g_optimizer = optim.RMSprop(generator.parameters(), lr=g_lr)
    # s_optimizer = optim.RMSprop(student.parameters(), lr=s_lr)
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    s_optimizer = optim.Adam(student.parameters(), lr=s_lr, betas=(0.5, 0.999))


    start_time = time.time()
    print("\n--- Starting PATE-GAN Training ---")
    print(f"Epochs: {epochs}, G LR: {g_lr}, S LR: {s_lr}, Laplace Scale: {lap_scale}")

    for epoch in range(epochs):
        generator.train()
        student.train()
        g_epoch_loss = 0.0
        s_epoch_loss = 0.0
        s_real_epoch_loss = 0.0
        s_fake_epoch_loss = 0.0
        num_batches = 0

        for real_x, _ in real_loader: # We only need real features X here
            real_x = real_x.to(device)
            batch_size = real_x.size(0)
            num_batches += 1

            # === Train Student ===
            s_optimizer.zero_grad()

            # 1. On Real Data with Noisy Labels from Teachers
            # Ensure teachers list is not empty
            if not teachers:
                 raise ValueError("Teacher ensemble is empty!")
            real_labels_noisy = noisy_aggregate_multiclass(
                teachers, real_x, num_classes=K, lap_scale=lap_scale, device=device
            )
            student_output_real = student(real_x)
            loss_s_real = F.cross_entropy(student_output_real, real_labels_noisy)

            # 2. On Fake Data Generated by G
            z = torch.randn(batch_size, noise_dim, device=device)
            # Generate labels randomly for the generator input
            gen_labels_cond = torch.randint(0, K, (batch_size,), device=device)
            # Detach fake data to avoid grads flowing back to G during S update
            fake_data = generator(z, gen_labels_cond).detach()
            # Label for fake data is K (the extra class)
            fake_labels_target = torch.full((batch_size,), K, dtype=torch.long, device=device)
            student_output_fake = student(fake_data)
            loss_s_fake = F.cross_entropy(student_output_fake, fake_labels_target)

            # Combine student losses and update
            loss_s = loss_s_real + loss_s_fake
            loss_s.backward()
            s_optimizer.step()

            # === Train Generator ===
            # Train Generator more frequently? Sometimes needed.
            # for _ in range(g_steps_per_d_step): # e.g., g_steps_per_d_step = 2
            g_optimizer.zero_grad()

            # Generate fresh fake data
            z = torch.randn(batch_size, noise_dim, device=device)
            gen_labels_cond = torch.randint(0, K, (batch_size,), device=device)
            gen_data = generator(z, gen_labels_cond)

            # We want the student to classify the generated data with the *original condition label*
            student_output_gen = student(gen_data)
            loss_g = F.cross_entropy(student_output_gen, gen_labels_cond)

            # Update generator
            loss_g.backward()
            g_optimizer.step()

            # Accumulate losses for logging
            g_epoch_loss += loss_g.item()
            s_epoch_loss += loss_s.item()
            s_real_epoch_loss += loss_s_real.item()
            s_fake_epoch_loss += loss_s_fake.item()

        # --- Epoch Logging ---
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            avg_g_loss = g_epoch_loss / num_batches
            avg_s_loss = s_epoch_loss / num_batches
            avg_s_real_loss = s_real_epoch_loss / num_batches
            avg_s_fake_loss = s_fake_epoch_loss / num_batches
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}] | Time: {elapsed_time:.2f}s")
            print(f"  G Loss: {avg_g_loss:.4f}")
            print(f"  S Loss (Total): {avg_s_loss:.4f} | S Real Loss: {avg_s_real_loss:.4f} | S Fake Loss: {avg_s_fake_loss:.4f}")

    print("--- PATE-GAN Training Finished ---")
    return generator, student

# ── SYNTHETIC DATA GENERATION ───────────────────────────────────────────────────
def generate_synthetic_data(generator: ConditionalGenerator,
                            num_samples: int,
                            noise_dim: int,
                            num_classes: int,
                            batch_size: int = 128,
                            device: torch.device = torch.device('cpu')) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data using the trained generator."""
    generator.to(device)
    generator.eval()
    X_syn_list, y_syn_list = [], []
    print(f"\n--- Generating {num_samples} synthetic samples ---")

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            if current_batch_size <= 0: break

            z = torch.randn(current_batch_size, noise_dim, device=device)
            # Generate synthetic labels uniformly
            cond = torch.randint(0, num_classes, (current_batch_size,), device=device)

            synthetic_data = generator(z, cond)

            X_syn_list.append(synthetic_data.cpu().numpy())
            y_syn_list.append(cond.cpu().numpy())

    X_syn = np.vstack(X_syn_list)
    y_syn = np.concatenate(y_syn_list)
    print(f"Generated X_syn shape: {X_syn.shape}, y_syn shape: {y_syn.shape}")
    return X_syn, y_syn

# ── TSTR EVALUATION ─────────────────────────────────────────────────────────────
def evaluate_tstr(X_syn: np.ndarray, y_syn: np.ndarray,
                  X_real_test: np.ndarray, y_real_test: np.ndarray,
                  scaler: MinMaxScaler):
    """Trains models on synthetic data and evaluates on real test data (TSTR)."""
    print("\n--- TSTR Evaluation ---")

    # Inverse scale synthetic data if models expect original range (optional, depends on model)
    # X_syn_inv = scaler.inverse_transform(X_syn)
    # X_real_test_inv = scaler.inverse_transform(X_real_test)
    # For simplicity, we'll use the scaled data [-1, 1] for both training and testing here.
    # Ensure the real test data is also scaled.
    X_real_test_scaled = X_real_test # Assuming X_real_test was already scaled

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=CONFIG['seed']),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=CONFIG['seed']),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=CONFIG['seed'], early_stopping=True),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=CONFIG['seed']),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name} on synthetic data...")
        start_time = time.time()
        try:
            # Check if synthetic data is valid
            if X_syn is None or y_syn is None or len(X_syn) == 0 or len(y_syn) == 0:
                 print(f"  Skipping {name}: Invalid synthetic data.")
                 results[name] = None
                 continue
            if len(np.unique(y_syn)) < 2:
                 print(f"  Skipping {name}: Synthetic data contains only one class.")
                 results[name] = None
                 continue

            model.fit(X_syn, y_syn)
            print(f"  Training finished in {time.time() - start_time:.2f}s")

            print(f"Evaluating {name} on real data...")
            preds = model.predict(X_real_test_scaled)
            acc = accuracy_score(y_real_test, preds)
            print(f"  {name} TSTR Accuracy: {acc:.4f}")
            results[name] = acc
        except Exception as e:
            print(f"  Error training/evaluating {name}: {e}")
            results[name] = None

    return results

# ── MAIN EXECUTION ──────────────────────────────────────────────────────────────
def main():
    """Main function to run the PATE-GAN pipeline."""
    print("--- Starting PATE-GAN Pipeline ---")
    # Ensure CONFIG uses the updated values
    current_config = CONFIG
    print(f"Using configuration: {current_config}")


    # Set random seeds for reproducibility
    np.random.seed(current_config['seed'])
    torch.manual_seed(current_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_config['seed'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Load & Preprocess Data ---
    try:
        df = pd.read_csv(current_config['data_path']) # [cite: 1]
    except FileNotFoundError:
        print(f"Error: Data file not found at {current_config['data_path']}")
        return

    if current_config['target_col'] not in df.columns:
        print(f"Error: Target column '{current_config['target_col']}' not found in the dataframe.") # [cite: 1] referenced via path
        return

    y = df[current_config['target_col']].values.astype(np.int64) # [cite: 1]
    X_df = df.drop(columns=[current_config['target_col']]) # [cite: 1]

    # Handle categorical features using one-hot encoding
    print("Performing one-hot encoding...")
    X_df = pd.get_dummies(X_df, drop_first=True) # drop_first avoids multicollinearity
    X = X_df.values.astype(np.float32)
    print(f"Data shape after encoding: X={X.shape}, y={y.shape}") # [cite: 1] approx shape

    # Scale features to [-1, 1] range (important for Tanh activation in Generator)
    print("Scaling data to [-1, 1]...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    K = len(np.unique(y)) # Number of classes in the target variable
    current_config['num_classes'] = K # Store K in config for later use
    data_dim = X_scaled.shape[1]
    print(f"Number of classes (K): {K}, Data dimension: {data_dim}")

    # --- Teacher Setup & Training ---
    print("\n--- Setting up and Training Teachers ---")
    data_splits = partition_data(X_scaled, y, current_config['n_teachers'])
    teachers: List[Teacher] = []
    teacher_loaders: List[DataLoader] = [] # Store loaders for evaluation

    if not data_splits:
         print("Error: No data splits created for teachers. Check partitioning logic or n_teachers.")
         return

    for i, (Xs, ys) in enumerate(data_splits):
        print(f"Training Teacher {i+1}/{len(data_splits)} on partition size {len(Xs)}...")
        if len(Xs) == 0:
             print(f"  Skipping Teacher {i+1} due to empty data partition.")
             continue
        loader = create_data_loader(Xs, ys, batch_size=current_config['batch_size'])
        # Need a valid loader to evaluate later, even if teacher is skipped? No, only add if teacher is trained.
        teacher = Teacher(input_dim=data_dim, num_classes=K, hidden_dim=current_config['t_hidden_dim'])
        teacher.train_teacher(loader, lr=current_config['teacher_lr'], epochs=current_config['teacher_epochs'], device=device)
        # Only add teacher and loader if training occurred
        teachers.append(teacher)
        teacher_loaders.append(loader)


    if not teachers:
         print("Error: No teachers were trained successfully. Aborting.")
         return
    print(f"\nTrained {len(teachers)} teachers.")


    # Optional: Evaluate Teacher Accuracy (helps diagnose issues)
    if current_config['eval_teachers'] and teachers:
        print("\n--- Evaluating Teacher Accuracy (on their own data partitions) ---")
        total_acc = 0
        if len(teachers) != len(teacher_loaders):
             print("Warning: Mismatch between number of teachers and loaders for evaluation. Skipping evaluation.")
        else:
            for i, teacher in enumerate(teachers):
                 # Use the corresponding loader saved earlier
                acc = teacher.evaluate(teacher_loaders[i], device=device)
                print(f"  Teacher {i+1} Accuracy: {acc:.2f}%")
                total_acc += acc
            if teachers: # Avoid division by zero if no teachers were trained
                print(f"  Average Teacher Accuracy: {total_acc / len(teachers):.2f}%")


    # --- PATE-GAN Model Initialization ---
    print("\n--- Initializing Generator and Student ---")
    generator = ConditionalGenerator(
        noise_dim=current_config['noise_dim'],
        data_dim=data_dim,
        num_classes=K,
        hidden_dim=current_config['g_hidden_dim']
    )
    # Student needs K real classes + 1 fake class
    student = Student(
        input_dim=data_dim,
        num_classes=K + 1,
        hidden_dim=current_config['s_hidden_dim']
    )
    print(generator)
    print(student)

    # Create DataLoader for the *entire* real dataset for PATE-GAN training
    # Note: Labels are not directly used in the loop but needed for DataLoader structure
    full_real_loader = create_data_loader(X_scaled, y, batch_size=current_config['batch_size'], shuffle=True)

    # --- Train PATE-GAN ---
    generator, student = train_pate_gan(
        generator, student, teachers, full_real_loader, current_config, device
    )

    # --- Generate Synthetic Data ---
    N_synthetic = X_scaled.shape[0] # Generate same number of samples as real data
    X_syn, y_syn = generate_synthetic_data(
        generator, N_synthetic, current_config['noise_dim'], K, current_config['batch_size'], device
    )

    # --- Evaluate Synthetic Data ---
    # Evaluate on the original scaled real data (X_scaled, y)
    evaluate_tstr(X_syn, y_syn, X_scaled, y, scaler)

    print("\n--- PATE-GAN Pipeline Finished ---")

if __name__ == '__main__':
    main()