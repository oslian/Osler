"""Source-only affine normalization for un-clipped native event tensors."""
import numpy as np
import torch


def patient_centers(samples, n_state):
    """One source summary per patient/variable; prefer the actual anchor label.

    For variables without an anchor label, use the median of observed history
    rows in this patient's stored examples. This is a sampling summary, not a
    deduplicated raw event ledger or a measurement-frequency estimate.
    """
    result = np.full(n_state, np.nan)
    observed = [[] for _ in range(n_state)]
    for example in samples.values():
        tokens = example['tokens'].detach().cpu().numpy()
        if tokens.ndim != 2 or tokens.shape[1] < 3*n_state:
            raise ValueError('Native variable/mask/age blocks required')
        mask = tokens[:, n_state:2*n_state]
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('Binary observation masks required')
        for col in np.flatnonzero(mask.any(axis=0)):
            values = tokens[mask[:, col].astype(bool), col]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite observed input')
            observed[col].extend(values.tolist())
    for col, values in enumerate(observed):
        if values:
            result[col] = np.median(values)
    for col, example in samples.items():
        value = float(example['truth_normalized'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite source label')
        result[col] = value
    return result


def fit_scaler(centers, train_rows):
    """Held-out values cannot affect location, scale or supported columns."""
    selected = np.asarray(centers, dtype=float)[train_rows]
    if selected.ndim != 2 or not len(selected):
        raise ValueError('Nonempty source matrix required')
    location = np.zeros(selected.shape[1]); scale = np.ones(selected.shape[1])
    counts = np.isfinite(selected).sum(axis=0)
    for col in np.flatnonzero(counts):
        values = selected[np.isfinite(selected[:, col]), col]
        location[col] = np.median(values)
        q25, q75 = np.quantile(values, [.25, .75])
        # Unit fallback is the existing normalized unit, not an imputed value.
        if q75 > q25:
            scale[col] = q75-q25
    return dict(location=location, scale=scale, counts=counts)


def transform_example(example, scaler):
    n = len(scaler['location']); col = example['target_index']
    if scaler['counts'][col] == 0:
        raise ValueError('Target has no source normalization support')
    x = example['tokens'].clone()
    location = torch.as_tensor(scaler['location'], dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scaler['scale'], dtype=x.dtype, device=x.device)
    supported = torch.as_tensor(scaler['counts']>0, device=x.device)
    mask = x[:, n:2*n].bool()
    if (mask & ~supported).any():
        raise ValueError('Observed variable has no source normalization support')
    x[:, :n] = torch.where(mask, (x[:, :n]-location)/scale, 0)
    return dict(example, tokens=x,
        truth_normalized=(example['truth_normalized']-scaler['location'][col])/scaler['scale'][col])


def inverse_value(value, col, scaler):
    return value*scaler['scale'][col]+scaler['location'][col]
