import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── DATA PARTITIONING ──────────────────────────────────────────────────────────
def partition_data(X, y, n_teachers):
    """Split dataset into n_teachers disjoint partitions."""
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
    """Create DataLoader for numpy arrays."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ── NOISY AGGREGATION ───────────────────────────────────────────────────────────
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

# ── MODEL CLASSES ───────────────────────────────────────────────────────────────
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, data_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        input_dim = noise_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
    def forward(self, x): return self.net(x)
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
    def forward(self, x): return self.net(x)

# ── TRAINING LOOP ──────────────────────────────────────────────────────────────
def train_pate_gan(generator, student, teachers, real_loader,
                   K, noise_dim, epochs=200, lap_scale=0.01,
                   lr=1e-4, device='cpu'):
    generator.to(device)
    student.to(device)
    g_opt = optim.Adam(generator.parameters(), lr=lr)
    s_opt = optim.Adam(student.parameters(), lr=lr)
    for epoch in range(epochs):
        for real_x, _ in real_loader:
            real_x = real_x.to(device)
            bsz = real_x.size(0)
            # ── student on real
            s_opt.zero_grad()
            real_lbls = noisy_aggregate_multiclass(
                teachers, real_x, num_classes=K, lap_scale=lap_scale, device=device
            )
            loss_real = F.cross_entropy(student(real_x), real_lbls)
            # ── student on fake
            z = torch.randn(bsz, noise_dim, device=device)
            cond = torch.randint(0, K, (bsz,), device=device)
            fake = generator(z, cond).detach()
            fake_lbl = torch.full((bsz,), K, dtype=torch.long, device=device)
            loss_fake = F.cross_entropy(student(fake), fake_lbl)
            (loss_real + loss_fake).backward()
            s_opt.step()
            # ── generator
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

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(42)
    torch.manual_seed(42)

    # Load & preprocess
    df = pd.read_csv('sampled_data.csv')
    y = df['readmitted'].values.astype(np.int64)
    X_df = df.drop(columns=['readmitted'])
    # one-hot encode any categorical features
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values.astype(np.float32)

    # scale to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_scaled = scaler.fit_transform(X)

    # Setup
    K = len(np.unique(y))             # =2
    n_teachers = 5                 # choose ≥2
    splits = partition_data(X_scaled, y, n_teachers)
    teachers = []
    for i, (Xs, ys) in enumerate(splits):
        t = Teacher(input_dim=X_scaled.shape[1], hidden_dim=256, num_classes=K)
        loader = create_data_loader(Xs, ys)
        t.train_teacher(loader, device=device)
        teachers.append(t)
    print(f"Trained {len(teachers)} teachers.")

    real_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled), torch.tensor(y)),
        batch_size=128, shuffle=True
    )

    # Init models
    noise_dim = 100
    gen = ConditionalGenerator(noise_dim, X_scaled.shape[1], K, hidden_dim=256)
    stu = Student(X_scaled.shape[1], hidden_dim=256, num_classes=K+1)

    # Train PATE-GAN
    gen, stu = train_pate_gan(
        gen, stu, teachers, real_loader,
        K, noise_dim,
        epochs=200, lap_scale=0.01,
        lr=1e-4, device=device
    )

    # Generate synthetic & TSTR
    N = X_scaled.shape[0]
    X_syn, y_syn = [], []
    batch = 128
    for i in range((N + batch - 1)//batch):
        size = min(batch, N - i*batch)
        z = torch.randn(size, noise_dim, device=device)
        cond = torch.randint(0, K, (size,), device=device)
        with torch.no_grad():
            out = gen(z, cond).cpu().numpy()
        X_syn.append(out)
        y_syn.append(cond.cpu().numpy())
    X_syn = np.vstack(X_syn)
    y_syn = np.concatenate(y_syn)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_syn, y_syn)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"TSTR Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
