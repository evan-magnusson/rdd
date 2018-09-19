import pandas as pd

def optimal_bandwidth(Y, X):
    # http://www.nber.org/papers/w14726.pdf
    # Step 1
    h1 = 1.84 * X.std() * (X.shape[0]**(-.2))
    Nh1neg = X[(X < 0) & (X > -h1)].shape[0]
    Nh1pos =X[(X >= 0) & (X < h1)].shape[0]
    Ybarh1neg = Y[(X < 0) & (X > -h1)].mean()
    Ybarh1pos = Y[(X >= 0) & (X < h1)].mean()
    fXc = (Nh1neg + Nh1pos) / (2 * X.shape[0] * h1)
    sig2c = (((Y[(X < 0) & (X > -h1)]-Ybarh1neg)**2).sum() +((Y[(X >= 0) & (X < h1)]-Ybarh1pos)**2).sum()) / (Nh1neg + Nh1pos)
    
    # Step 2
    medXneg = X[X<0].median()
    medXpos = X[X>=0].median()
    dat_temp = pd.DataFrame({'Y': Y,'X':X, 'idx_col':X.index})
    dat_temp = dat_temp.loc[(dat_temp['X'] >= medXneg) & (dat_temp['X'] <= medXpos)]
    dat_temp['treat'] = 0
    dat_temp.loc[dat_temp['X'] >= 0, 'treat'] = 1
    dat_temp['X2'] = X**2
    dat_temp['X3'] = X**3
    eqn = 'Y ~ 1 + treat + X + X2 + X3'
    results = smf.ols(eqn, data=dat_temp).fit()
    m3 = 6*results.params.loc['X3']
    h2pos = 3.56 * (X[X>=0].shape[0]**(-1/7.0)) * (sig2c/(fXc * np.max([m3**2, .01]))) ** (1/7.0)
    h2neg = 3.56 * (X[X<0].shape[0]**(-1/7.0)) * (sig2c/(fXc * np.max([m3**2, .01]))) ** (1/7.0)
    Yplus = Y[(X>=0) & (X<=h2pos)]
    Xplus = X[(X>=0) & (X<=h2pos)]
    dat_temp = pd.DataFrame({'Y': Yplus,'X':Xplus, 'idx_col':Xplus.index})
    dat_temp['X2'] = X**2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2pos = 2*results.params.loc['X2']
    Yneg = Y[(X<0) & (X>=-h2neg)]
    Xneg = X[(X<0) & (X>=-h2neg)]
    dat_temp = pd.DataFrame({'Y': Yneg,'X':Xneg, 'idx_col':Xneg.index})
    dat_temp['X2'] = X**2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2neg = 2*results.params.loc['X2']
    
    # Step 3
    rpos = 720*sig2c / (X[(X>=0) & (X<=h2pos)].shape[0] * h2pos**4)
    rneg = 720*sig2c / (X[(X<0) & (X>=-h2neg)].shape[0] * h2neg**4)
    CK = 3.4375
    hopt = CK * (2*sig2c/(fXc * ((m2pos - m2neg)**2 + (rpos+rneg))))**.2 * Y.shape[0]**(-.2)
    
    return hopt