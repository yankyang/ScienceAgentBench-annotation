import os, json, random
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

DATA = Path("./benchmark/datasets/deephit/synthetic_comprisk.csv")
OUT = Path("./pred_results"); OUT.mkdir(exist_ok=True, parents=True)
GOLD = Path("./benchmark/eval_programs/gold_results"); GOLD.mkdir(exist_ok=True, parents=True)
torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)

class LabTrans(LabTransDiscreteTime):
    def transform(self, d, e):
        d, f = super().transform(d, e > 0)
        e[f == 0] = 0
        return d, e.astype("int64")

class Net(torch.nn.Module):
    def __init__(self, inf, nr, out):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(inf, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, nr * out)
        )
        self.nr, self.out = nr, out
    def forward(self, x):
        y = self.fc(x)
        return y.view(y.size(0), self.nr, self.out)

def simple_concordance(durations, events_any, scores):
    idx = np.argsort(durations)
    t, e, s = durations[idx], events_any[idx], scores[idx]
    num, den = 0, 0
    n = len(t)
    for i in range(n):
        if e[i] != 1:  
            continue
        for j in range(i+1, n):
            if t[j] == t[i]:  
                continue
            den += 1
            if s[i] > s[j]:
                num += 1
            elif s[i] == s[j]:
                num += 0.5
    return float(num / den) if den > 0 else 0.0

def main():
    df = pd.read_csv(DATA)
    df_test = df.sample(frac=0.2, random_state=1234)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=1234)
    df_train = df_train.drop(df_val.index)

    x = lambda d: d.drop(columns=['time','label','true_time','true_label']).values.astype('float32')
    lab = LabTrans(5)
    get_y = lambda d: (d['time'].values, d['label'].values)
    ytr = lab.fit_transform(*get_y(df_train))
    yv = lab.transform(*get_y(df_val))
    dt, ev = get_y(df_test)
    pd.DataFrame({'duration': dt, 'event': ev}).to_csv(GOLD / "deephit_competing_risks_gold.csv", index=False)

    net = Net(x(df_train).shape[1], int(ytr[1].max()), len(lab.cuts))
    opt = tt.optim.Adam(lr=0.01)
    model = DeepHit(net, opt, alpha=0.2, sigma=0.1, duration_index=lab.cuts)
    model.fit(x(df_train), ytr, 128, 20, verbose=True, val_data=(x(df_val), yv))

    surv = model.predict_surv_df(x(df_test)).sort_index()
    risk = 1.0 - surv.iloc[-1].values.astype(np.float64)
    events_any = (ev != 0).astype(int)
    ctd = simple_concordance(dt.astype(np.float64), events_any, risk)

    fig, ax = plt.subplots(figsize=(6,4))
    surv.iloc[:, :3].plot(ax=ax)
    ax.grid(linestyle="--")
    plt.savefig(OUT / "deephit_competing_risks_cif.png", dpi=100)
    plt.close()

    metrics = {'concordance_td': ctd}
    with open(OUT / "deephit_competing_risks_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    main()
    os._exit(0)
