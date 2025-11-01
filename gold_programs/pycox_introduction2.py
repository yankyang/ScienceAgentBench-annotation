import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt

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

    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    idx_train, idx_test = idx[:split], idx[split:]
    x_train, x_test = x[idx_train], x[idx_test]
    durations_train, durations_test = durations[idx_train], durations[idx_test]
    events_train, events_test = events[idx_train], events[idx_test]

    num_durations = 100
    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train = labtrans.fit_transform(durations_train, events_train)
    y_test = labtrans.transform(durations_test, events_test)

    train = (x_train, y_train)
    val = (x_test, y_test)

    in_features = x.shape[1]
    num_nodes = [32, 32]
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features=labtrans.out_features, batch_norm=True, dropout=0.1)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)

    batch_size = 256
    epochs = 10
    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(*train, batch_size, epochs, callbacks, val_data=val, verbose=False)

    surv = model.predict_surv_df(x_test)

    idx_points = np.linspace(0, len(surv.index)-1, num=5, dtype=int)
    times_to_save = surv.index[idx_points]
    surv_at_times = surv.loc[times_to_save].T

    result_df = pd.DataFrame(x_test, columns=cols_features)
    result_df['duration'] = durations_test
    result_df['event'] = events_test
    for t in times_to_save:
        result_df[f"surv_prob_t{t}"] = surv_at_times[t].values
    result_df.to_csv(out_dir / "pycox_introduction2_pred.csv", index=False)

    fig, ax = plt.subplots()
    surv.mean(axis=1).plot(ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    fig.savefig(out_dir / "pycox_introduction2_plot.png")

if __name__ == "__main__":
    main()
