import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchtuples as tt
import matplotlib.pyplot as plt
from pathlib import Path
from pycox.models import CoxPH

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    data_path = Path("./benchmark/datasets/pycox_support/support.csv")
    out_dir = Path("./pred_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    removed_idx = np.random.choice(df.index, size=5, replace=False)
    df = df.drop(index=removed_idx).reset_index(drop=True)
    cols_features = df.columns.difference(['duration', 'event'])
    x = df[cols_features].values.astype('float32')
    y = (df['duration'].values, df['event'].values)
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    idx_train, idx_test = idx[:split], idx[split:]
    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = (y[0][idx_train], y[1][idx_train]), (y[0][idx_test], y[1][idx_test])
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train[0]), torch.from_numpy(y_train[1]))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    in_features = x.shape[1]
    net = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    model = CoxPH(net, tt.optim.Adam)
    model.fit(x_train, y_train, batch_size=256, epochs=10, verbose=False)
    model.compute_baseline_hazards()
    risk_scores = model.predict(x_test)
    result_df = pd.DataFrame(x_test, columns=cols_features)
    result_df["duration"] = y_test[0]
    result_df["event"] = y_test[1]
    result_df["risk_score"] = risk_scores
    result_df.to_csv(out_dir / "pycox_introduction_pred.csv", index=False)
    surv = model.predict_surv_df(x_test)
    fig, ax = plt.subplots()
    surv.mean(axis=1).plot(ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    fig.savefig(out_dir / "pycox_introduction_plot.png")

if __name__ == "__main__":
    main()
