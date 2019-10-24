import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def optimal_bandwidth(Y, X, cut=0):
    '''
    DESCRIPTION:
        For a given outcome Y and running variable X, computes the optimal bandwidth
        h using a triangular kernel. For more information, see 
        "OPTIMAL BANDWIDTH CHOICE FOR THE REGRESSION DISCONTINUITY ESTIMATOR",
        by Imbens and Kalyanaraman, at http://www.nber.org/papers/w14726.pdf

    INPUTS:
        Two equal length pandas Series
            Y: the outcome variable
            X: the running variable
        cut: value for the threshold of the rdd (scalar) (default is 0)
    
    OUTPUTS:
        Scalar optimal bandwidth value
    '''

    assert(X.shape[0] == Y.shape[0], "X and Y are not of the same length")
    assert(np.sum(pd.isnull(X)) == 0, "NaNs are present in the running variable X")
    assert(np.sum(pd.isnull(Y)) == 0, "NaNs are present in the running variable X")


    # Normalize X
    X = X - cut

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
    dat_temp = pd.DataFrame({'Y': Y,'X':X})
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
    dat_temp = pd.DataFrame({'Y': Yplus,'X':Xplus})
    dat_temp['X2'] = X**2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2pos = 2*results.params.loc['X2']
    Yneg = Y[(X<0) & (X>=-h2neg)]
    Xneg = X[(X<0) & (X>=-h2neg)]
    dat_temp = pd.DataFrame({'Y': Yneg,'X':Xneg})
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


def truncated_data(data, xname, bandwidth=None, yname=None, cut=0):
    '''

    Drop observations from dataset that are outside 
        a given (or optimal) bandwidth

    INPUTS:
        data: data with the X and Y values (pandas DataFrame)
        xname: Name of your running variable (string)
        bandwidth: Bandwidth (scalar) (if None given, the optimal bandwidth is computed)
        yname: The name of your outcome variable (string) (only needed if no bandwidth is given)
        cut: The value of your threshold (scalar) (default is 0)

    OUTPUTS:
        pandas DataFrame with observations outside of the bandwidth dropped
    
    '''
    if bandwidth==None:
        if yname==None:
            raise NameError("You must supply either a bandwidth or the name of your outcome variable.")
        else:
            bandwidth = optimal_bandwidth(data[yname], data[xname], cut=cut)
    data_new = data.loc[np.abs(data[xname]-cut)<=bandwidth, ]
    return data_new


def rdd(input_data, xname, yname=None, cut=0, interact_treatment_running=False, equation=None, controls=None, noconst=False, weights=1, verbose=True):
    '''
    This function implements a linear regression (ordinary or weighted least squares can be used) for 
        the estimation of regressing the outcome variable on the running variable.  A "TREATED" variable
        is created, the coefficient on which is the causal effect of being to the right of the threshold.

        The user may specify a list of controls to be added linearly, or supply their own equation.  

    INPUT:
        input_data: dataset with outcome and running variables (and potentially controls) (pandas DataFrame)
        xname: name of running variable (string)
        yname: name of outcome variable (string) (default is None - not needed if you include your own equation)
        cut: location of threshold in xname (scalar) (default is 0)
        equation: Estimation equation as a string (see Statsmodels formula syntax for more info)
        controls: List of controls to include in the estimation (list of strings) (not needed if you include your own equation)
        noconst: If True, model does not estimate an intercept (bool) (default is false)
        weights: Weights for weighted least squares (numpy array) (default is equal weights, ie OLS)

    OUTPUT:
        Statsmodels object

    '''
    if yname==None and equation==None:
        raise NameError("You must supply either a outcome variable name or an equation to estimate.")
    if 'TREATED' in input_data.columns:
        raise NameError("TREATED is a reserved column name.  Please change the name.")
    data = input_data.copy() # To avoid SettingWithCopy warnings
    data['TREATED'] = np.where(data[xname] >= cut, 1, 0)
    if equation==None:
        equation = yname + ' ~ TREATED + ' + xname
        if interact_treatment_running == True:
            interact_term = ' + TREATED:' + xname
            equation += add_term
        if controls != None:
            equation_controls = ' + '.join(controls)
            equation += ' + ' + equation_controls
    if noconst==True:
        equation += ' -1'
    if verbose==True:
        print('Estimation Equation:\t', equation)
    rdd_model = smf.wls(equation, data=data, weights=weights)
    return rdd_model


def bin_data(data, yname, xname, bins=50, agg_fn=np.mean):
    '''
    When datasets are so large that traditional RDD scatter plots are difficult to read, 
        this will group observations by their X values into a set number of bins and compute
        the mean outcome value in that bin.  

    INPUT:
        data: dataset (pandas DataFrame)
        yname: Name of outcome variable (string)
        xname: Name of running variable (string)
        bins: Desired number of bins to group data by (integer) (default is 50)

    OUTPUT:
        A pandas DataFrame that has a row for each bin with columns:
            yname: The average value of the outcome variable in that bin
            xname: the midpoint value of the running variable in that bin
            n_obs: The number of observations in this bin
            
    '''
    hist, edges = np.histogram(data[xname], bins=bins)
    bin_midpoint = np.zeros(edges.shape[0]-1)
    binned_df = pd.DataFrame(np.zeros((edges.shape[0]-1, 1)))
    for i in range(edges.shape[0]-1):
        bin_midpoint[i] = (edges[i] + edges[i+1]) / 2
        if i < edges.shape[0]-2:
            dat_temp = data.loc[(data[xname] >= edges[i]) & (
                data[xname] < edges[i+1]), :]
            binned_df.loc[binned_df.index[i], yname] = agg_fn(dat_temp[yname])
            binned_df.loc[binned_df.index[i], xname] = bin_midpoint[i]
            binned_df.loc[binned_df.index[i], 'n_obs'] = dat_temp.shape[0]
        else:
            dat_temp = data.loc[(data[xname] >= edges[i]) & (
                data[xname] <= edges[i+1]), :]
            binned_df.loc[binned_df.index[i], yname] = agg_fn(dat_temp[yname])
            binned_df.loc[binned_df.index[i], xname] = bin_midpoint[i]
            binned_df.loc[binned_df.index[i], 'n_obs'] = dat_temp.shape[0]
    return binned_df
