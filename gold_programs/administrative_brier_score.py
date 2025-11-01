# administrative_brier_score.py

import pandas as pd
import torchtuples as tt
from pycox.models import LogisticHazard
from pycox.evaluation import brier_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import numpy as np
import os
from sklearn.model_selection import train_test_split

def main():
    # Step 1: Load CSV data
    df = pd.read_csv("benchmark/datasets/sac_admin5/sac_admin5.csv")

    # Step 2: Extract features and labels
    feature_cols = [f"x{i}" for i in range(23)]
    x = df[feature_cols].values.astype("float32")
    durations = df["duration"].values
    events = df["event"].values

    # Step 3: Train/val/test split
    x_train, x_temp, d_train, d_temp, e_train, e_temp = train_test_split(
        x, durations, events, test_size=0.4, random_state=42)
    x_val, x_test, d_val, d_test, e_val, e_test = train_test_split(
        x_temp, d_temp, e_temp, test_size=0.5, random_state=42)

    # Step 4: Standardize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Step 5: Label transformation
    from pycox.models import LogisticHazard
    labtrans = LogisticHazard.label_transform(10)
    y_train = labtrans.fit_transform(d_train, e_train)
    y_val = labtrans.transform(d_val, e_val)

    # Step 6: Define model
    net = torch.nn.Sequential(
        nn.Linear(x_train.shape[1], 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, labtrans.out_features)
    )
    model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.optimizer.set_lr(0.01)

    # Step 7: Train model
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=512,
              callbacks=[tt.callbacks.EarlyStopping()],
              val_data=(x_val, y_val))

    # Step 8: Predict survival function
    surv = model.predict_surv_df(x_test)

    # Step 9: Evaluate with Brier Score (直接用brier_score函数，不用EvalSurv)
    times = np.arange(1, int(d_test.max()) + 1)
    # brier_score需要 survival prob, durations, events
    scores = []
    for t in times:
        scores.append(brier_score(surv, d_test, e_test, t))
    brier_scores = np.array(scores)

    # Step 10: Save results
    os.makedirs("./pred_results", exist_ok=True)
    out_df = pd.DataFrame({
        "time": times,
        "brier_score": brier_scores
    })
    out_df.to_csv("./pred_results/administrative_brier_score.csv", index=False)
    print("✅ Brier scores saved to ./pred_results/administrative_brier_score.csv")

if __name__ == "__main__":
    main()
