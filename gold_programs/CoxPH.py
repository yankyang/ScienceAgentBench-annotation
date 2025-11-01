import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt

import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items


def main():
    data_path = './benchmark/datasets/cox/metabric.csv'
    output_dir = './pred_results'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['duration', 'event'])
    df['duration'] = df['duration'].astype(float)
    df['event'] = df['event'].astype(int)

    np.random.seed(1234)
    torch.manual_seed(123)

    df_test = df.sample(frac=0.2, random_state=42)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=0.2, random_state=24)
    df_train = df_train.drop(df_val.index)


    remove_n = 5
    drop_idx = df_train.sample(n=remove_n, random_state=999).index
    df_train = df_train.drop(drop_idx)

    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    get_target = lambda df: (
        df['duration'].values.astype('float32'),
        df['event'].values.astype('float32')
    )
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = (x_val, y_val)

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, 1, batch_norm=True, dropout=0.1, output_bias=False
    )

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)

    epochs = 64
    callbacks = [tt.callbacks.EarlyStopping()]
    model.fit(x_train, y_train, batch_size=256, epochs=epochs,
              callbacks=callbacks, verbose=False, val_data=val)

    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    concordance = float(ev.concordance_td())
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ibs = float(ev.integrated_brier_score(time_grid))
    inbll = float(ev.integrated_nbll(time_grid))

    result = {
        "concordance": concordance,
        "integrated_brier_score": ibs,
        "integrated_nbll": inbll,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test)
    }

    output_path = os.path.join(output_dir, 'coxph_results.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print("CoxPH evaluation complete. Results saved to", output_path)


if __name__ == "__main__":
    main()
