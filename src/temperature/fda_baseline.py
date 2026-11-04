import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

FEVER_C = 37.5

def metrics(err):
    err = pd.Series(err)
    return dict(bias=err.mean(), MAE=err.abs().mean(), RMSE=float(np.sqrt((err ** 2).mean())))

def facial_feature(df, prefix):
    cols = [f"{prefix}{k}" for k in range(1, 5) if f"{prefix}{k}" in df.columns]
    return df[cols].mean(axis=1)

def cv_calibrate(x, y, seed=42):
    pred = cross_val_predict(LinearRegression(), x.to_frame(), y,
                             cv=KFold(5, shuffle=True, random_state=seed))
    return pd.Series(pred, index=y.index)

def bootstrap_ci(values, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    return np.percentile([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)], [2.5, 97.5])

def screening(pred, y, thr):
    pos, flag = y >= FEVER_C, pred >= thr
    tp, fn = (pos & flag).sum(), (pos & ~flag).sum()
    tn, fp = (~pos & ~flag).sum(), (~pos & flag).sum()
    return dict(n_pos=int(pos.sum()), sensitivity=float(tp / max(tp + fn, 1)),
                specificity=float(tn / max(tn + fp, 1)))
