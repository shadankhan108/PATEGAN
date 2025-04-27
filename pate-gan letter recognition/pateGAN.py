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
def partition_data(X, y, n_teachers):
    """
    Split dataset into n_teachers disjoint partitions.
    """
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
    """
    Create DataLoader for numpy arrays.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def noisy_aggregate_multiclass(teacher_ensemble, x_batch, num_classes, lap_scale=0.01, device='cpu'):
    """
    Aggregate teacher votes with Laplace noise.
    """
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


class ConditionalGenerator(nn.Module):
    """Generator with conditional input."""
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
            nn.Tanh()  # because data scaled to [-1,1]
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([z, one_hot], dim=1)
        return self.net(x)


class Teacher(nn.Module):
    """Simple MLP for classification."""
    def __init__(self, input_dim, hidden_dim=256, num_classes=26):
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
    """Student with K+1 outputs for PATE-GAN."""
    def __init__(self, input_dim, hidden_dim=256, num_classes=27):
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

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.argmax(self(x), dim=1)


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
            # train student
            s_opt.zero_grad()
            real_lbls = noisy_aggregate_multiclass(
                teachers, real_x, num_classes=K, lap_scale=lap_scale, device=device)
            loss_real = F.cross_entropy(student(real_x), real_lbls)
            # fake data
            z = torch.randn(bsz, noise_dim, device=device)
            cond = torch.randint(0, K, (bsz,), device=device)
            fake = generator(z, cond).detach()
            fake_lbl = torch.full((bsz,), K, dtype=torch.long, device=device)
            loss_fake = F.cross_entropy(student(fake), fake_lbl)
            (loss_real + loss_fake).backward()
            s_opt.step()
            # train generator
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





def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(42)
    torch.manual_seed(42)

    # load & preprocess
    df = pd.read_csv('letter.csv')
    le = LabelEncoder()
    df['letter'] = le.fit_transform(df['letter'])
    K = df['letter'].nunique()
    X = df.drop(columns=['letter']).values.astype(np.float32)
    y = df['letter'].values.astype(np.int64)

    # scale to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # train teachers
    n_teachers = 4
    splits = partition_data(X_scaled, y, n_teachers)
    teachers = []
    for Xp, yp in splits:
        t = Teacher(input_dim=X.shape[1], hidden_dim=256, num_classes=K)
        loader = create_data_loader(Xp, yp)
        t.train_teacher(loader, device=device)
        teachers.append(t)
    print(f"Trained {len(teachers)} teachers.")

    # full data loader
    ds = TensorDataset(torch.tensor(X_scaled), torch.tensor(y))
    real_loader = DataLoader(ds, batch_size=128, shuffle=True)

    # init models
    noise_dim = 20
    gen = ConditionalGenerator(noise_dim, X.shape[1], K, hidden_dim=256)
    stu = Student(X.shape[1], hidden_dim=256, num_classes=K+1)

    # train PATE-GAN
    gen, stu = train_pate_gan(gen, stu, teachers, real_loader,
                              K, noise_dim, epochs=200,
                              lap_scale=0.01, device=device)

    # generate synthetic data
    N = X_scaled.shape[0]
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

    # TSTR Evaluation
    models = {
    'Logistic Regression': LogisticRegression(max_iter=100, verbose=1),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # more trees
        max_depth=10,      # allow deeper trees
        min_samples_split=2,
        max_features='sqrt',
        random_state=42,
        verbose=1
    ),
    'XGBoost': xgb.XGBClassifier(
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1  # xgboost uses 'verbosity' not 'verbose'
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # Deeper and wider
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=30,
        random_state=42,
        verbose=True
    )
    }

    for name, model in models.items():
        model.fit(X_syn, y_syn)
        preds = model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        print(f"TSTR Accuracy with {name}: {acc:.4f}")

if __name__ == '__main__':
    main()