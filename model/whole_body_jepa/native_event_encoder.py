"""Native event histories using the existing single predict-update filter."""
import numpy as np
from osler_jepa.patient_state import NeuralBeliefFilter


class NativeEventFilter(NeuralBeliefFilter):
    def forward(self,observations,observation_mask=None):
        if observation_mask is None:
            raise ValueError('explicit native lab observation mask required')
        # Six event features precede the clock; last is observed-rate mask.
        active=observation_mask.bool()|observations[...,-2].bool()
        return super().forward(observations,active)


def native_arrays(base,native,n_state,module_count):
    """Keep query context exact; use padded ragged event histories only."""
    old_rows=base['query_rows'];qptr=native['query_ptr'];ids=native['history_indices']
    if len(qptr)!=len(old_rows)+1:raise ValueError('query count mismatch')
    if qptr[0]!=0 or qptr[-1]!=len(ids) or (np.diff(qptr)<0).any():raise ValueError('invalid query pointers')
    tokens=native['input'];n=len(tokens)
    if len(ids) and (ids.min()<0 or ids.max()>=n):raise ValueError('invalid event indices')
    rows=n+1+np.arange(len(old_rows));width=int(np.diff(qptr).max())
    x=np.concatenate([np.zeros((1,tokens.shape[1]),np.float32),tokens,base['input'][old_rows]])
    history=np.zeros((len(x),width),np.int32)
    for i,row in enumerate(rows):
        seq=ids[qptr[i]:qptr[i+1]]+1
        if len(seq):history[row,-len(seq):]=seq
    result=dict(input=x,current=x[:,:n_state].copy(),mask=x[:,n_state:2*n_state].copy(),
                ages=x[:,2*n_state:3*n_state].copy(),presence=x[:,3*n_state+1:3*n_state+1+module_count].copy(),
                history=history,horizon=np.full(len(x),6.,np.float32))
    for key in ('current','mask','ages','presence','horizon'):
        np.testing.assert_array_equal(result[key][rows],base[key][old_rows])
    values=np.concatenate([np.full((1,3),np.nan),native['event_values'],np.full((len(rows),3),np.nan)])
    return result,rows,values
