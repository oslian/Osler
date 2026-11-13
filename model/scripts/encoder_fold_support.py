"""Keep the predictor's observation vocabulary fixed by its training support."""
import torch
from scripts.encoder_fold_scaling import transform_example


def supported_example(example, scaler):
    """Unsupported observations remain outside this fold's learned input path."""
    n=len(scaler['counts'])
    clean=example['tokens'].clone()
    unsupported=torch.as_tensor(scaler['counts']==0,device=clean.device)
    clean[:,:n][:,unsupported]=0
    clean[:,n:2*n][:,unsupported]=0
    clean[:,2*n:3*n][:,unsupported]=7
    keep=clean[:,n:2*n].bool().any(-1)
    if not keep.any():
        raise ValueError('No source-supported independent history remains')
    # The anatomical observation encoder reconstructs presence from these masks.
    # Remove empty events so unseen variables cannot enter through event timing.
    return transform_example(dict(example,tokens=clean[keep],
        event_hours=example['event_hours'][keep]),scaler)
