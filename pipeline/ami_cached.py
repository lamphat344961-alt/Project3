"""Exact hypergeometric EMI lookup for repeated fixed-support AMI evaluation.

No Monte Carlo, tail cutoff, asymptotic approximation or changed normalizer.
The lookup stores each contingency cell's expected MI contribution conditional
on its two marginal counts and n. All feasible positive overlap counts are used.
"""
import numpy as np
from scipy.special import gammaln


def expected_cell_table(n, observed_counts):
    counts=np.unique(np.asarray(observed_counts,dtype=np.int64))
    counts=counts[(counts>0)&(counts<=n)]
    if not len(counts):
        return np.zeros((1,1),dtype=np.float64)
    maximum=int(counts.max())
    table=np.full((maximum+1,maximum+1),np.nan,dtype=np.float64)
    table[0,:]=table[:,0]=0.
    factorial=gammaln(np.arange(n+1,dtype=np.float64)+1)
    for a in counts:
        bs=counts[counts>=a]
        for start in range(0,len(bs),128):
            b=bs[start:start+128]
            if a==n:
                values=np.zeros(len(b))
            else:
                k=np.arange(1,a+1,dtype=np.int64)[None,:]
                remaining=n-a-b[:,None]+k
                valid=remaining>=0
                constant=factorial[a]+factorial[b]+factorial[n-a]+factorial[n-b]-factorial[n]
                logp=constant[:,None]-factorial[k]-factorial[a-k]-factorial[b[:,None]-k]-factorial[np.maximum(remaining,0)]
                probability=np.exp(logp)
                probability[~valid]=0.
                term=(k/n)*(np.log(k*n)-np.log(a)-np.log(b[:,None]))
                values=np.sum(term*probability,axis=1)
                values[b==n]=0.
            table[a,b]=table[b,a]=values
    return table


def entropy_counts(counts,n):
    positive=np.asarray(counts,dtype=np.float64)
    positive=positive[positive>0]
    return float(-np.sum((positive/n)*np.log(positive/n)))


def score_pair(a,b,table,counts_a=None,counts_b=None,k=49):
    """Return arithmetic-normalized AMI and ARI for integer labels in [0,k)."""
    n=len(a)
    if n==0:
        return np.nan,np.nan
    if len(b)!=n:
        raise ValueError('Label supports differ')
    contingency=np.bincount(np.asarray(a,dtype=np.int64)*k+b,minlength=k*k).reshape(k,k)
    ca=contingency.sum(axis=1) if counts_a is None else counts_a
    cb=contingency.sum(axis=0) if counts_b is None else counts_b
    nonempty_a=int(np.count_nonzero(ca))
    nonempty_b=int(np.count_nonzero(cb))
    # A bijection between nonempty contingency rows/columns is perfect agreement.
    if np.all(np.count_nonzero(contingency,axis=1)<=1) and np.all(np.count_nonzero(contingency,axis=0)<=1):
        return 1.,1.
    if nonempty_a==1 or nonempty_b==1:
        ami=0.
    else:
        rows,cols=np.nonzero(contingency)
        values=contingency[rows,cols].astype(np.float64)
        mi=max(0.,float(np.sum((values/n)*(np.log(values*n)-np.log(ca[rows])-np.log(cb[cols])))))
        contributions=table[ca[:,None],cb[None,:]]
        if not np.isfinite(contributions).all():
            raise ValueError('EMI lookup does not cover observed marginal counts')
        emi=float(contributions.sum())
        normalizer=(entropy_counts(ca,n)+entropy_counts(cb,n))/2
        denominator=normalizer-emi
        numerator=mi-emi
        eps=np.finfo(np.float64).eps
        denominator=min(denominator,-eps) if denominator<0 else max(denominator,eps)
        numerator=min(numerator,-eps) if numerator<0 else max(numerator,eps)
        ami=numerator/denominator
    choose=lambda x: np.sum(np.asarray(x,dtype=np.float64)*(np.asarray(x)-1)/2)
    total=n*(n-1)/2
    observed=choose(contingency)
    aa,bb=choose(ca),choose(cb)
    expected=aa*bb/total if total else 0.
    denominator=(aa+bb)/2-expected
    ari=(observed-expected)/denominator if denominator else 1.
    return float(ami),float(ari)
