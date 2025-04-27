import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# --- Utilities ---
def partition_data(X, y, n_teachers):
    data_splits = []
    N = len(X)
    split_size = N // n_teachers
    indices = np.random.permutation(N)
    for i in range(n_teachers):
        start = i * split_size
        end = (i + 1) * split_size if i < n_teachers - 1 else N
        idx = indices[start:end]
        data_splits.append((X[idx], y[idx]))
    return data_splits

def create_data_loader(X, y, batch_size=128):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def noisy_aggregate_multiclass(teacher_ensemble, x_batch, num_classes, lap_scale=0.01, device='cpu'):
    x_batch = x_batch.to(device)
    all_votes = []
    for teacher in teacher_ensemble:
        teacher.eval()
        with torch.no_grad():
            preds = torch.argmax(teacher(x_batch), dim=1)
            all_votes.append(preds.cpu().numpy())
    all_votes = np.array(all_votes).T
    agg_labels = []
    for votes_i in all_votes:
        counts = np.bincount(votes_i, minlength=num_classes).astype(np.float32)
        noisy = counts + np.random.laplace(0, lap_scale, size=num_classes)
        agg_labels.append(np.argmax(noisy))
    return torch.tensor(agg_labels, dtype=torch.long, device=device)

# --- Models ---
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, data_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([z, one_hot], dim=1)
        return self.net(x)

class Teacher(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def train_teacher(self, loader, lr=1e-3, epochs=15, device='cpu'):
        self.to(device)
        opt = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            self.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self(xb), yb).backward()
                opt.step()

class Student(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- Training Loop ---
def train_pate_gan(generator, student, teachers, real_loader,
                   K, noise_dim, epochs=200, lap_scale=0.01,
                   lr=1e-4, device='cuda'):
    generator.to(device)
    student.to(device)
    g_opt = optim.Adam(generator.parameters(), lr=lr)
    s_opt = optim.Adam(student.parameters(), lr=lr)
    for epoch in range(epochs):
        for real_x, _ in real_loader:
            real_x = real_x.to(device)
            bsz = real_x.size(0)

            # Train Student
            s_opt.zero_grad()
            real_lbls = noisy_aggregate_multiclass(teachers, real_x, num_classes=K, lap_scale=lap_scale, device=device)
            loss_real = F.cross_entropy(student(real_x), real_lbls)

            z = torch.randn(bsz, noise_dim, device=device)
            cond = torch.randint(0, K, (bsz,), device=device)
            fake = generator(z, cond).detach()
            fake_lbl = torch.full((bsz,), K, dtype=torch.long, device=device)
            loss_fake = F.cross_entropy(student(fake), fake_lbl)

            (loss_real + loss_fake).backward()
            s_opt.step()

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(bsz, noise_dim, device=device)
            cond = torch.randint(0, K, (bsz,), device=device)
            gen_data = generator(z, cond)
            g_loss = F.cross_entropy(student(gen_data), cond)
            g_loss.backward()
            g_opt.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | S-loss: {(loss_real+loss_fake).item():.4f} | G-loss: {g_loss.item():.4f}")

    return generator, student

# --- Main Execution ---
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(42)
    torch.manual_seed(42)

    # Load dataset
    df = pd.read_csv('loan.csv')
    le = LabelEncoder()
    df['Online'] = le.fit_transform(df['Online'])
    y = df['Online'].values.astype(np.int64)

    feature_cols = [col for col in df.columns if col != 'Online']
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]

    # Process features
    df_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

    X_processed = pd.concat([df_numerical, df_categorical], axis=1)
    X = X_processed.values.astype(np.float32)

    # Update K
    K = np.unique(y).shape[0]

    # Teacher training
    n_teachers = 4
    splits = partition_data(X, y, n_teachers)
    teachers = []
    for Xp, yp in splits:
        t = Teacher(input_dim=X.shape[1], hidden_dim=256, num_classes=K)
        loader = create_data_loader(Xp, yp)
        t.train_teacher(loader, device=device)
        teachers.append(t)
    print(f"Trained {len(teachers)} teachers.")

    real_loader = create_data_loader(X, y)

    noise_dim = 20
    gen = ConditionalGenerator(noise_dim, X.shape[1], K, hidden_dim=256)
    stu = Student(X.shape[1], hidden_dim=256, num_classes=K+1)

    # Train
    gen, stu = train_pate_gan(gen, stu, teachers, real_loader,
                              K, noise_dim, epochs=200,
                              lap_scale=0.01, device=device)

    # Generate synthetic data
    N = X.shape[0]
    X_syn, y_syn = [], []
    batch = 128
    for i in range((N + batch - 1) // batch):
        size = min(batch, N - i * batch)
        z = torch.randn(size, noise_dim, device=device)
        cond = torch.randint(0, K, (size,), device=device)
        with torch.no_grad():
            out = gen(z, cond).cpu().numpy()
        X_syn.append(out)
        y_syn.append(cond.cpu().numpy())
    X_syn = np.vstack(X_syn)
    y_syn = np.concatenate(y_syn)

    # TSTR evaluation
    models = {
        'Logistic Regression': LogisticRegression(max_iter=100, verbose=1),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, verbose=1),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', n_estimators=200, max_depth=10, random_state=42, verbosity=1),
        'MLP': MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=30, random_state=42, verbose=True)
    }

    for name, model in models.items():
        model.fit(X_syn, y_syn)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"TSTR Accuracy with {name}: {acc:.4f}")

if __name__ == '__main__':
    main()
