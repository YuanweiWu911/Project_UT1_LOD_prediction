def ARforecast_dXdY(dX, mjd, ic, max_lag, pred_num, p0):
    """Forecasts dX and dY using autoregressive modeling.

    Args:
        dX (pd.Series): Time series of dX or dY values
        mjd (pd.Series): Modified Julian Dates corresponding to dX values
        ic (str): Information criterion for AR model selection ('aic', 'bic', etc.)
        max_lag (int): Maximum lag order to consider for AR model
        pred_num (int): Number of days to forecast
        p0 (list): Initial parameters for least squares fitting

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast dates
            - Series_new_dX (pd.Series): Extended dX series including forecast values
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    pred_days = next_n_days(end_day,pred_num)
    mjd_pred_days = Time(pred_days.tolist()).mjd
    mjd_pred_days = pd.Series(mjd_pred_days,index=pred_days)
   
    ##################################
    #  lsq_fit of the PMX and PMY
    lsq_res,x_fit = lsq_func_dXdY(dX,mjd,p0)
    
    # lsq_fit extrapolate n days
    lsq_fit = func_dXdY(lsq_res, mjd_pred_days)
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = dX-x_fit
    ###################
    #forecast

    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag)
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag,ic=ic)
    pred_lsq_resid=ARmodel.predict(end=pred_days[-1])
    #AR_pred + lsq_expolate
    delta = pred_lsq_resid[pred_days]+lsq_fit

    #append prediction
    Series_new_dX = dX.append(delta)
#   print(Series_new_pmxy)
    new_mjd = mjd.append(mjd_pred_days)
    p=len(ARmodel.params)
#   print(new_mjd)
    
    return new_mjd, Series_new_dX, p
 


def ARforecast_pmxy_1(pmxy, mjd, ic, max_lag):
    """Forecasts polar motion (pmx/pmy) for the next day using autoregressive modeling.

    Args:
        pmxy (pd.Series): Time series of polar motion values (pmx or pmy)
        mjd (pd.Series): Modified Julian Dates corresponding to pmxy values
        ic (str): Information criterion for AR model selection ('aic', 'bic', etc.)
        max_lag (int): Maximum lag order to consider for AR model

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast date
            - Series_new_pmxy (pd.Series): Extended pmxy series including forecast value
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    end_day_tmr = next_day(end_day)
    mjd_end_day_tmr = Time(end_day_tmr).mjd
    mjd_end_day_tmr = pd.Series(mjd_end_day_tmr,index=[end_day_tmr])
   
    ##################################
    #  lsq_fit of the UT1R/LODR series
    lsq_res,x_fit = lsq_func_pmxy(pmxy,mjd)
    
    # lsq_fit extrapolate 1 more day
    lsq_fit = func_pmxy(lsq_res,np.arange(mjd_end_day_tmr,mjd_end_day_tmr+1,1))
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = pmxy-x_fit
    ###################
    #forecast

    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag) # p = 88
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag,ic=ic) # p = 88

    pred_lsq_resid=ARmodel.predict(end=end_day_tmr)
    
    delta = pred_lsq_resid[end_day_tmr]+lsq_fit
    twr_pmxy = pd.Series(delta,index=[end_day_tmr])
    
    New_pmxy = pd.Series(twr_pmxy,index=[end_day_tmr])
    Series_new_pmxy = pmxy.append(New_pmxy)
    new_mjd = mjd.append(mjd_end_day_tmr)
    p=len(ARmodel.params)
    
    return new_mjd, Series_new_pmxy, p

def ARforecast_pmxy_2(pmxy, mjd, ic, max_lag, pred_num):
    """Forecasts polar motion (pmx/pmy) for multiple days using autoregressive modeling.

    Args:
        pmxy (pd.Series): Time series of polar motion values (pmx or pmy)
        mjd (pd.Series): Modified Julian Dates corresponding to pmxy values
        ic (str): Information criterion for AR model selection ('aic', 'bic', etc.)
        max_lag (int): Maximum lag order to consider for AR model
        pred_num (int): Number of days to forecast

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast dates
            - Series_new_pmxy (pd.Series): Extended pmxy series including forecast values
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    pred_days = next_n_days(end_day,pred_num)
    mjd_pred_days = Time(pred_days.tolist()).mjd
    mjd_pred_days = pd.Series(mjd_pred_days,index=pred_days)
   
    ##################################
    #  lsq_fit of the PMX and PMY
    lsq_res,x_fit = lsq_func_pmxy(pmxy,mjd)
    
    # lsq_fit extrapolate n days
    lsq_fit = func_pmxy(lsq_res, mjd_pred_days)
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = pmxy-x_fit
    ###################
    #forecast

    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag)
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag,ic=ic)
    pred_lsq_resid=ARmodel.predict(end=pred_days[-1])
    #AR_pred + lsq_expolate
    delta = pred_lsq_resid[pred_days]+lsq_fit

    #append prediction
    Series_new_pmxy = pmxy.append(delta)
#   print(Series_new_pmxy)
    new_mjd = mjd.append(mjd_pred_days)
    p=len(ARmodel.params)
#   print(new_mjd)
    
    return new_mjd, Series_new_pmxy, p

def ARforecast_pmxy_3(pmxy, mjd, ic, max_lag, pred_num):
    """Forecasts polar motion (pmx/pmy) for multiple days using autoregressive modeling.

    Args:
        pmxy (pd.Series): Time series of polar motion values (pmx or pmy)
        mjd (pd.Series): Modified Julian Dates corresponding to pmxy values
        ic (str): Information criterion for AR model selection ('aic', 'bic', etc.)
        max_lag (int): Maximum lag order to consider for AR model
        pred_num (int): Number of days to forecast

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast dates
            - Series_new_pmxy (pd.Series): Extended pmxy series including forecast values
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    pred_days = next_n_days(end_day,pred_num)
    mjd_pred_days = Time(pred_days.tolist()).mjd
    mjd_pred_days = pd.Series(mjd_pred_days,index=pred_days)
   
    ##################################
    #  lsq_fit of the PMX and PMY
    lsq_res,x_fit = lsq_func_pmxy(pmxy,mjd)
    
    # lsq_fit extrapolate n days
    lsq_fit = func_pmxy(lsq_res, mjd_pred_days)
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = pmxy-x_fit
    ###################
    #forecast

    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag)
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag,ic=ic)
    pred_lsq_resid=ARmodel.predict(end=pred_days[-1])
    #AR_pred + lsq_expolate
    delta = pred_lsq_resid[pred_days]+lsq_fit

    #append prediction
    Series_new_pmxy = pmxy.append(delta)
#   print(Series_new_pmxy)
    new_mjd = mjd.append(mjd_pred_days)
    p=len(ARmodel.params)
#   print(new_mjd)
    
    return new_mjd, Series_new_pmxy, p

def ARforecast_ut1_1(sut1, mjd, ic, max_lag):
    """Forecasts UT1-UTC for the next day using autoregressive modeling.

    Args:
        sut1 (pd.Series): Time series of UT1-UTC values
        mjd (pd.Series): Modified Julian Dates corresponding to sut1 values
        ic (str): Information criterion for AR model selection ('aic', 'bic', etc.)
        max_lag (int): Maximum lag order to consider for AR model

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast date
            - Series_new_sut1 (pd.Series): Extended sut1 series including forecast value
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    end_day_tmr = next_day(end_day)
    mjd_end_day_tmr = Time(end_day_tmr).mjd
    mjd_end_day_tmr = pd.Series(mjd_end_day_tmr,index=[end_day_tmr])
   
    # remove zonal tidal for UT1, LOD and Omega
    (dut1,dlod,domega) = calc_dut1(mjd)
    #lodr= lod - dlod
    sut1r=sut1-dut1
    dsut1r = sut1r.diff(1)
#   ddsut1r = dsut1r.diff(1)
    ##################################
    #  lsq_fit of the UT1R/LODR series
    lsq_res,x_fit = lsq_func_ut1(dsut1r[1:],mjd[1:])
    
    # lsq_fit extrapolate 1 more day
    lsq_fit = func_ut1(lsq_res,np.arange(mjd_end_day_tmr,mjd_end_day_tmr+1,1))
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = dsut1r[1:]-x_fit
    ###################
    #forecast

    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag) # p = 88
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag,ic=ic) # p = 88

    pred_lsq_resid=ARmodel.predict(end=end_day_tmr)
    
    delta = pred_lsq_resid[end_day_tmr]+lsq_fit
    twr_dsut1r = pd.Series(delta,index=[end_day_tmr])
    
    new_dsut1r = dsut1r.append(twr_dsut1r)
    new_sut1r = new_dsut1r[end_day_tmr]+sut1r[end_day]
    
    #add tidal term
    (dut1_tmr,dlod_tmr,domega_tmr) = calc_dut1([mjd_end_day_tmr])
    new_sut1 = dut1_tmr + new_sut1r
    
    New_sut1 = pd.Series(new_sut1,index=[end_day_tmr])
    Series_new_sut1 = sut1.append(New_sut1)
    new_mjd = mjd.append(mjd_end_day_tmr)
    p=len(ARmodel.params)
    
    return new_mjd, Series_new_sut1, p

def ARforecast_ut1_2(sut1, mjd, ic, max_lag):
    """Forecasts UT1-UTC for the next day using autoregressive modeling with second differences.

    Args:
        sut1 (pd.Series): Time series of UT1-UTC values
        mjd (pd.Series): Modified Julian Dates corresponding to sut1 values
        ic (str): Information criterion for AR model selection ('aic', 'bic', 'hqic', or 't-stat')
        max_lag (int): Maximum lag order to consider for AR model

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast date
            - Series_new_sut1 (pd.Series): Extended sut1 series including forecast value
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    end_day_tmr = next_day(end_day)
    mjd_end_day_tmr = Time(end_day_tmr).mjd
    mjd_end_day_tmr = pd.Series(mjd_end_day_tmr,index=[end_day_tmr])
   
    # remove zonal tidal for UT1, LOD and Omega
    (dut1,dlod,domega) = calc_dut1(mjd)
    #lodr= lod - dlod
    sut1r=sut1-dut1
    dsut1r = sut1r.diff(1)
    ddsut1r = dsut1r.diff(1)
    ##################################
    #  lsq_fit of the UT1R/LODR series
    lsq_res,x_fit = lsq_func_ut1(ddsut1r[2:],mjd[2:])
    
    # lsq_fit extrapolate 1 more day
    lsq_fit = func_ut1(lsq_res,np.arange(mjd_end_day_tmr,mjd_end_day_tmr+1,1))
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = ddsut1r[2:]-x_fit
    ###################
    #forecast
#   ARmodel = sm.tsa.AR(endog=lsq_resid).fit() fastest by p = 42 fixed
    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag)
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag, ic=ic)
#   ARmodel = sm.tsa.AR(endog=lsq_resid).fit(maxlag=200, ic='t-stat') #p = 88 faster than aic
#   ARmodel = sm.tsa.AR(endog=lsq_resid).fit(ic='aic',trend='nc',method = 'cmle',disp=-1)
    pred_lsq_resid=ARmodel.predict(end=end_day_tmr)
    
    delta = pred_lsq_resid[end_day_tmr]+lsq_fit
    twr_ddsut1r = pd.Series(delta,index=[end_day_tmr])
    
    new_ddsut1r = ddsut1r.append(twr_ddsut1r)
    new_sut1r = new_ddsut1r[end_day_tmr]+dsut1r[end_day]+sut1r[end_day]
    
    #add tidal term
    (dut1_tmr,dlod_tmr,domega_tmr) = calc_dut1([mjd_end_day_tmr])
    new_sut1 = dut1_tmr + new_sut1r
    
    New_sut1 = pd.Series(new_sut1,index=[end_day_tmr])
    Series_new_sut1 = sut1.append(New_sut1)
    new_mjd = mjd.append(mjd_end_day_tmr)
    p=len(ARmodel.params)
    
    return new_mjd, Series_new_sut1, p

def ARforecast_ut1_3(sut1, mjd, ic, max_lag, pred_num):
    """Forecasts UT1-UTC for multiple days using autoregressive modeling with second differences.

    Args:
        sut1 (pd.Series): Time series of UT1-UTC values
        mjd (pd.Series): Modified Julian Dates corresponding to sut1 values
        ic (str): Information criterion for AR model selection ('aic', 'bic', 'hqic', or 't-stat')
        max_lag (int): Maximum lag order to consider for AR model
        pred_num (int): Number of days to forecast

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast dates
            - Series_new_sut1 (pd.Series): Extended sut1 series including forecast values
            - p (int): Number of parameters in selected AR model
    """

    endday = mjd.index[-1]
    pred_days = next_n_days(endday,pred_num)
#   print(pred_days)
    mjd_pred_days = Time(pred_days.tolist()).mjd
    mjd_pred_days = pd.Series(mjd_pred_days,index=pred_days)
    
    # remove zonal tidal for UT1, LOD and Omega
    (dut1,dlod,domega) = calc_dut1(mjd)  #unit sec
    dut1 = dut1*1E6  #unit 1E-6 sec
    print("dut1 range: %f %f"%(min(dut1),max(dut1)))
    print("sut1 range: %f %f"%(min(sut1),max(sut1)))
    
    sut1r=sut1-dut1   #unit micro sec
    dsut1r = sut1r.diff(1)
    ddsut1r = dsut1r.diff(1)
    ##################################
    #  lsq_fit of the UT1R/LODR series
    lsq_res,x_fit = lsq_func_ut1(ddsut1r[2:],mjd[2:])
    
    # lsq_fit extrapolate 1 more day
    lsq_fit = func_ut1(lsq_res,mjd_pred_days)
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = ddsut1r[2:]-x_fit
    ###################
    #forecast
    if ic=='':
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag)
    else:
        ARmodel = sm.tsa.AR(endog=lsq_resid).fit(max_lag = max_lag, ic=ic)
    pred_lsq_resid=ARmodel.predict(end=pred_days[-1])
    
    delta = pred_lsq_resid[pred_days]+lsq_fit
    pred_ddsut1r = pd.Series(delta,index=pred_days)
    pred_dsut1r = pred_ddsut1r.cumsum()+dsut1r[-1]
    pred_sut1r = pred_dsut1r.cumsum()+sut1r[-1]

#   #add tidal term
    (dut1_pred,dlod_pred,domega_tmr) = calc_dut1(mjd_pred_days) #unit sec
    dut1_pred = dut1_pred*1E6              #unit micro sec
    new_sut1 = dut1_pred + pred_sut1r  #unit micro sec

    Series_new_sut1 = sut1.append(new_sut1)
    new_mjd = mjd.append(mjd_pred_days)
    p=len(ARmodel.params)

    return new_mjd, Series_new_sut1, p

def _ARforecast_ut1_3(sut1, p, mjd):
    """Forecasts UT1-UTC for the next day using autoregressive modeling with second differences (internal version).

    Args:
        sut1 (pd.Series): Time series of UT1-UTC values
        p (int): Number of parameters for AR model
        mjd (pd.Series): Modified Julian Dates corresponding to sut1 values

    Returns:
        tuple: Contains:
            - new_mjd (pd.Series): Extended MJD series including forecast date
            - Series_new_sut1 (pd.Series): Extended sut1 series including forecast value
            - p (int): Number of parameters in selected AR model
    """

    end_day = mjd.index[-1]
    end_day_tmr = next_day(end_day)
    mjd_end_day_tmr = Time(end_day_tmr).mjd
    mjd_end_day_tmr = pd.Series(mjd_end_day_tmr,index=[end_day_tmr])
   
    # remove zonal tidal for UT1, LOD and Omega
    (dut1,dlod,domega) = calc_dut1(mjd)
    print("dut1 range: %f %f"%(min(dut1),max(dut1)))
    print("sut1 range: %f %f"%(min(sut1),max(sut1)))
    #lodr= lod - dlod
    sut1r=sut1-dut1
    dsut1r = sut1r.diff(1)
    ddsut1r = dsut1r.diff(1)
    ##################################
    #  lsq_fit of the UT1R/LODR series
    lsq_res,x_fit = lsq_func_ut1(ddsut1r[2:],mjd[2:])
    
    # lsq_fit extrapolate 1 more day
    lsq_fit = func_ut1(lsq_res,np.arange(mjd_end_day_tmr,mjd_end_day_tmr+1,1))
    
    #lsq residuals as inputs of ARmodel 
    lsq_resid = ddsut1r[2:]-x_fit
    ###################
    #forecast use ARMA instead of AR
    #################################################################################
#   ARmodel = sm.tsa.AR(endog=lsq_resid).fit()
#   pred_lsq_resid=ARmodel.predict(end=end_day_tmr)
    p_s = []
    aic_s = []
    bic_s = []
    hqic_s = []
    ARmodels = []
    fcsts = []
    for p in np.arange(1,20,1):
        (aic,bic,hqic,fcst) = ar_forecast(lsq_resid, p)
        p_s.append(p)
        aic_s.append(aic)
        bic_s.append(bic)
        hqic_s.append(hqic)
        fcsts.append(fcst)
    best_indx = np.where(aic_s == min(aic_s))
    best_indx = best_indx[0][0]
    best_fcst=fcsts[best_indx]
#   print("best p = %i"%p_s[best_indx])
#   print("best fcst = %i"%fcsts[best_indx])
    ##################################################################################
    
    delta = best_fcst+lsq_fit
    twr_ddsut1r = pd.Series(delta,index=[end_day_tmr])
    
    new_ddsut1r = ddsut1r.append(twr_ddsut1r)
    new_sut1r = new_ddsut1r[end_day_tmr]+dsut1r[end_day]+sut1r[end_day]
    
    #add tidal term
    (dut1_tmr,dlod_tmr,domega_tmr) = calc_dut1([mjd_end_day_tmr])
    new_sut1 = dut1_tmr + new_sut1r
    
    New_sut1 = pd.Series(new_sut1,index=[end_day_tmr])
    Series_new_sut1 = sut1.append(New_sut1)
    new_mjd = mjd.append(mjd_end_day_tmr)
    
    return new_mjd, Series_new_sut1, p_s[best_indx]

def _func_dPsidEps_res(arg, t, y):
    """Calculates residuals for dPsi/dEpsilon model fitting.

    Args:
        arg (list): Model parameters to be determined by least squares fitting
        t (float): Modified Julian Date in days
        y (float): Observed dUT1R or LODR value

    Returns:
        float: Residual between model prediction and observed value
    """
    a = arg[0]
    b = arg[1]
    c = arg[2]
    d = arg[3]
    e = arg[4]
    f = arg[5]
    g = arg[6]
    m = arg[7]
    n = arg[8]
    t1 = arg[9]
    t2 = arg[10]
    t3 = arg[11]

    p1 = np.float64(t1)
    p2 = np.float64(t2)
    p3 = np.float64(t3)
    residual = (a + b * t + c * t * t
              + d * np.cos(2.0 * np.pi * t / p1)
              + e * np.sin(2.0 * np.pi * t / p1)
              + f * np.cos(2.0 * np.pi * t / p2)
              + g * np.sin(2.0 * np.pi * t / p2)
              + m * np.cos(2.0 * np.pi * t / p3)
              + n * np.sin(2.0 * np.pi * t / p3)) - y

    return residual

def _func_dXdY_res(arg, t, y):
    """Calculates residuals for dX/dY model fitting.

    Args:
        arg (list): Model parameters to be determined by least squares fitting
        t (float): Modified Julian Date in days
        y (float): Observed dUT1R or LODR value

    Returns:
        float: Residual between model prediction and observed value
    """
    a = arg[0]
    b = arg[1]
    c = arg[2]
    d = arg[3]
    e = arg[4]
    f = arg[5]
    g = arg[6]
    t1 = arg[7]
    t2 = arg[8]

    p1 = np.float64(t1)
    p2 = np.float64(t2)
    residual = a + b * t + c * t * t + \
        d * np.cos(2.0 * np.pi * t / p1) + \
        e * np.sin(2.0 * np.pi * t / p1) + \
        f * np.cos(2.0 * np.pi * t / p2) + \
        g * np.sin(2.0 * np.pi * t / p2) - y
    return residual
