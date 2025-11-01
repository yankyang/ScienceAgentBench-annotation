import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import torchtuples as tt
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
    durations = df['duration'].values
    events = df['event'].values

    # Train/test split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    idx_train, idx_test = idx[:split], idx[split:]
    x_train, x_test = x[idx_train], x[idx_test]
    durations_train, durations_test = durations[idx_train], durations[idx_test]
    events_train, events_test = events[idx_train], events[idx_test]

    y_train = (durations_train, events_train)
    y_test = (durations_test, events_test)

    in_features = x.shape[1]

    # MLP 架构
    mlp_net = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    model_mlp = CoxPH(mlp_net, tt.optim.Adam)
    model_mlp.fit(x_train, y_train, batch_size=256, epochs=10, verbose=False)
    model_mlp.compute_baseline_hazards()

    cnn_net = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features*16, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    x_train_cnn = x_train[:, None, :]
    x_test_cnn = x_test[:, None, :]
    model_cnn = CoxPH(cnn_net, tt.optim.Adam)
    model_cnn.fit(x_train_cnn, y_train, batch_size=256, epochs=10, verbose=False)
    model_cnn.compute_baseline_hazards()

    risk_scores_mlp = model_mlp.predict(x_test).flatten()
    risk_scores_cnn = model_cnn.predict(x_test_cnn).flatten()

    result_df = pd.DataFrame(x_test, columns=cols_features)
    result_df["duration"] = durations_test
    result_df["event"] = events_test
    result_df["risk_score_mlp"] = risk_scores_mlp
    result_df["risk_score_cnn"] = risk_scores_cnn
    result_df.to_csv(out_dir / "pycox_networks_pred.csv", index=False)

    surv_mlp = model_mlp.predict_surv_df(x_test)
    surv_cnn = model_cnn.predict_surv_df(x_test_cnn)

    fig, ax = plt.subplots()
    surv_mlp.mean(axis=1).plot(ax=ax, label="MLP")
    surv_cnn.mean(axis=1).plot(ax=ax, label="CNN")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    ax.legend()
    fig.savefig(out_dir / "pycox_networks_plot.png")

if __name__ == "__main__":
    main()
