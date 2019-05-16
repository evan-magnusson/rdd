import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

'''
Additional functions to create:
    - RDD Tests
        - distribution plot
        - bin test
        - other mccrary tests?
        - balance
        - randomness reg test
        - continuity plots
        - continuity regs
    - Run RDD (verbose or non-verbose) (controls) (different functional forms of poly)
    - rdd plots
'''

def optimal_bandwidth(Y, X):
    '''
    DESCRIPTION:
        For a given outcome Y and running variable X, computes the optimal bandwidth
        h. For more information, see "OPTIMAL BANDWIDTH CHOICE FOR THE REGRESSION 
        DISCONTINUITY ESTIMATOR", by Imbens and Kalyanaraman, at
        http://www.nber.org/papers/w14726.pdf

    INPUTS:
        Two equal length pandas series

    OUTPUTS:

    TODO: 
        - accept numpy arrays
        - accept an x that doesn't have the thresh at 0 - if they give the threshold
        - remove these idx_col? 
        - remove any pandas dependencies?
        - what should change if I want controls

    '''

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


def truncated_data(data, xname, bandwidth):
    '''
    To Do:
        - remove pandas dependencies?
        - allow an option, if no bandwidth given, to do optimal bandwidth
        - is it strict inequality on bandwidth?
    '''
    data_new = data.loc[data[xname]<=bandwidth, ]
    return data_new


def rdd(input_data, yname, xname):
    '''
    To Do:
        - instead of equation, allow for default yname ~ xname + c
        - allow for it just to be data, yname, and xname, and a toggle for finding the optimal bandwidth
        - or allow for it to be data, yname, xname, and a input for your own bandwidth
        - allow for a list of controls
        - allow for weighted least squares
        - should I not call this rdd?
        - allow it to give something with the 'treated' binary already?
        - allow for noconst
        - allow for someone to already have a treated column
        - better way to make the treated column?
        - anything different than this output? i don't want to output the summary, in case people want params
    '''
    data = input_data.copy() # To avoid SettingWithCopy warnings
    data['treated'] = np.where(data[xname] >= 0, 1, 0)
    equation = yname + ' ~ treated + ' + xname
    rdd_model = smf.ols(equation, data=data)
    rdd_results = rdd_model.fit()
    return rdd_results

