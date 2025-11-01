import os, json, random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.datasets import metabric
from pycox.models import PMF
from pycox.evaluation import EvalSurv

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

DATA = Path("./benchmark/datasets/cox/metabric.csv")
OUT = Path("./pred_results"); OUT.mkdir(parents=True, exist_ok=True)
GOLD = Path("./benchmark/eval_programs/gold_results"); GOLD.mkdir(parents=True, exist_ok=True)

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
def simple_concordance(durations, events, risk_scores):
    idx = np.argsort(durations)
    t = durations[idx].astype(float)
    e = events[idx].astype(int)
    s = risk_scores[idx].astype(float)
    num = 0.0
    den = 0.0
    n = len(t)
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(i + 1, n):
            if t[j] == t[i]:
                continue
            den += 1.0
            if s[i] > s[j]:
                num += 1.0
            elif s[i] == s[j]:
                num += 0.5
    return float(num / den) if den > 0 else 0.0

def main():
    df = metabric.read_df()
    df_test = df.sample(frac=0.2, random_state=1234)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=1234)
    df_train = df_train.drop(df_val.index)

    cols_standardize = ['x0','x1','x2','x3','x8']
    cols_leave = ['x4','x5','x6','x7']
    standardize = [([c], StandardScaler()) for c in cols_standardize]
    leave = [(c, None) for c in cols_leave]
    mapper = DataFrameMapper(standardize + leave)
    x_train = mapper.fit_transform(df_train).astype('float32')
    x_val = mapper.transform(df_val).astype('float32')
    x_test = mapper.transform(df_test).astype('float32')

    num_durations = 10
    labtrans = PMF.label_transform(num_durations)
    get_target = lambda d: (d['duration'].values, d['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    durations_test, events_test = get_target(df_test)

    pd.DataFrame({'duration': durations_test, 'event': events_test}).to_csv(
        GOLD / "pmf_metabric_gold.csv", index=False
    )

    net = tt.practical.MLPVanilla(
        in_features=x_train.shape[1],
        num_nodes=[32, 32],
        out_features=labtrans.out_features,
        batch_norm=True,
        dropout=0.1
    )
    model = PMF(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.optimizer.set_lr(0.01)
    model.fit(x_train, y_train, batch_size=256, epochs=20, verbose=True, val_data=(x_val, y_val))

    surv = model.interpolate(10).predict_surv_df(x_test)
    fig, ax = plt.subplots()
    surv.iloc[:, :5].plot(drawstyle='steps-post', ax=ax)
    ax.set_ylabel("S(t | x)")
    ax.set_xlabel("Time")
    ax.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT / "pmf_metabric_surv.png", dpi=100)
    plt.close(fig)

    risk = 1.0 - surv.iloc[-1].values.astype(np.float64)
    metrics = {"concordance_td": simple_concordance(durations_test, events_test, risk)}

    with open(OUT / "pmf_metabric_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training done. Metrics:", metrics)

if __name__ == "__main__":
    main()
    os._exit(0)
