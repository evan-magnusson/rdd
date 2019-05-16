import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

'''
Additional functions to create:
    - RDD Tests
        - bin test
        - other mccrary tests?
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
    data_new = data.loc[np.abs(data[xname])<=bandwidth, ]
    return data_new


def rdd(input_data, xname, yname=None, cut=0, equation=None, controls=None, noconst=False):
    '''
    To Do:
        - return an error if yname AND equation are empty
        - return an error if 'TREATED' is in the column (unless you're using your own equation)
        - allow for it just to be data, yname, and xname, and a toggle for finding the optimal bandwidth
        - or allow for it to be data, yname, xname, and a input for your own bandwidth
        - allow for weighted least squares
        - allow it to give a different treatment column
        - return just teh ols object, and let them do standard errors and stuff?
        - anything different than this output? i don't want to output the summary, in case people want params
    Notes:
        - more complex controls requires you to put in your own equation
        - say what TREATED is
    '''
    data = input_data.copy() # To avoid SettingWithCopy warnings
    data['TREATED'] = np.where(data[xname] >= cut, 1, 0)
    if equation==None:
        equation = yname + ' ~ TREATED + ' + xname
        if controls != None:
            equation_controls = ' + '.join(controls)
            equation += ' + ' + equation_controls
    if noconst==True:
        equation += ' -1'
    rdd_model = smf.ols(equation, data=data)
    rdd_results = rdd_model.fit()
    return rdd_results


def bin_data(data, yname, xname, bins):
    '''
    To Do:
        - this could take full or cut data
        - there is likely a much more efficient way to do this with groupby or .where()
    '''
    hist, edges = np.histogram(data[xname], bins=bins)
    bin_midpoint = np.zeros(edges.shape[0]-1)
    binned_df = pd.DataFrame(np.zeros((edges.shape[0]-1, 1)))
    for i in range(edges.shape[0]-1):
        bin_midpoint[i] = (edges[i] + edges[i+1]) / 2
        if i < edges.shape[0]-2:
            dat_temp = data.loc[(data[xname] >= edges[i]) & (
                data[xname] < edges[i+1]), :]
            binned_df.loc[binned_df.index[i], yname] = dat_temp[yname].mean()
            binned_df.loc[binned_df.index[i], xname] = bin_midpoint[i]
            binned_df.loc[binned_df.index[i], 'n_obs'] = dat_temp.shape[0]
        else:
            dat_temp = data.loc[(data[xname] >= edges[i]) & (
                data[xname] <= edges[i+1]), :]
            binned_df.loc[binned_df.index[i], yname] = dat_temp[yname].mean()
            binned_df.loc[binned_df.index[i], xname] = bin_midpoint[i]
            binned_df.loc[binned_df.index[i], 'n_obs'] = dat_temp.shape[0]
    return binned_df
