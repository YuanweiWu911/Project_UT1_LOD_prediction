# miscfunc.py
#
# useful functions used in iers pacakge
# The expressions for the fundamental arguments of nutation are given by 
# IERS convention 2010, Section 5.7.2
#
# writer: yuanwei Wu @NTSC 2019-08-14
# version 1.0 2019-08-14
#             2019-08-15  sut1_utc, tested
#             2019-08-16  extrap, diff, acumu, tested
#                         powersp faster than powersp2 
#                         powersp2, psd unit in second
#                         lsq_func_ut1
#             2019-08-16  np.argwhere, reshape index
##########################################################################
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import optimize
import statsmodels.api as sm
from astropy.time import Time
from .lunisolar import calc_dut1
##############################################################################


def get_time():
    """return the system time with format of hh:mm:ss on 2019/01/01"""

    t = [0, 1, 2, 3, 4, 5]
    t[0] = time.localtime()[0]
    t[0] = str(t[0])
    for i in np.arange(1, 6):
        t[i] = time.localtime()[i]
        if t[i] < 10:
            t[i] = '0' + str(t[i])
        else:
            t[i] = str(t[i])
    a = t[3] + ':' + t[4] + ':' + t[5] + ' on ' + t[0] + '/' + t[1] + '/' + t[2]
    return a


def d2jd(tstr):
    """year-mn-dy string to jd and mjd
    inputs:
        timestring  '2000-01-01'

    outputs:
        jd
        mjd
        yearmndy 20000101

    Note:
        toordinal function returns the number of days since December 31, 1 BC at 00:00
        Julian day is the number of day since January 1, 4713 BC at 12:00
        the difference is 1721424.5
    """
    fmt = '%Y-%m-%d'
    dt = datetime.strptime(tstr, fmt)
    tt = dt.timetuple()
    jd = dt.toordinal()+1721424.5
    mjd = jd - 2400000.5
    return jd, mjd, tt.tm_year * 1000 + tt.tm_yday


def mjd2jc(mjd):
    """modified julian day to julian century
    input: mjd
    output: julian century
    """
    mjd = (np.float64(mjd) - 51544.5) / 36525
    return mjd


def mjd2jd(mjd):
    """ modified julian day to julian day
    inputs: mjd
    outputs: jd
    """
    return mjd + 2400000.5


def jd2mjd(jd):
    """julian day to modified julian day
    inputs: mjd
    outputs: jd
    """
    return jd - 2400000.5


def sut1utc(ut1_utc):
    """ from disconnected ut1 series to a continous ut1_utc series
    input: 
        ut1_utc, list or np.ndarray
        series of ut1_utc within [-1,1] second
    output:
        ut1_utc, np.ndarry
        leap seconds added
    """
    ut1_utc = np.array(ut1_utc)
    sut1_utc = ut1_utc
    dut1_utc = ut1_utc[1:] - ut1_utc[0:-1]
    index = np.argwhere(dut1_utc > 0.9)
    index = np.reshape(index, len(index))
    for i in index:
        sut1_utc[i+1:] = sut1_utc[i+1:] - 1.0
    return sut1_utc


def extrap(y, order):
    """ extraplate list or np.ndarray at the beginning and end of array
    inputs:
          y, list or np.ndarray
          order, 1 linear
                 2 2nd order polynomial
                 3 cubic
    outputs:
          y, and 1 elements at the beginning and end of inputs
    """
    leny = len(y)
    x = range(0, leny)
    f = InterpolatedUnivariateSpline(x, y, k=order)
    result = np.insert(y, 0, f([-1]))
    result = np.append(result, f([leny]))
    return result


def diff(y, order=1):
    """difference a list or np.ndarray
    input:
        y, a list or np.array
        order = -1  backward
              = 1   forward
              = 0   mid
    output: difference of the list or array
    """
    # extraplate at the beginning and end
    y2 = extrap(y, 3)
    yp = np.zeros(len(y) - 1)
    if order == 0:
        yp = (y2[2:] - y2[0:-2]) / 2.0
    elif order == 1:
        yp = y2[2:] - y
    elif order == -1:
        yp = y-y2[:-2]
    return yp


def acumu(dy, y0, order=1):
    """
      input:
           diff results
           order = 1 forward diff/accumu
           order = -1 backward diff/accumu
      output:
           accumulate diff results to recover diff inputs
    """
    intdy = np.zeros(len(dy))
    if order == 1:
        dy = np.insert(dy, 0, 0)
        intdy = np.add.accumulate(dy) + y0
        intdy = intdy[:-1]  # remove last elements, due to forward diff
    elif order == -1:
        dy = dy[1:]
        dy = np.insert(dy, 0, 0)
        intdy = np.add.accumulate(dy) + y0

    return intdy


def powersp(x):
    """
    inputs: 
        x: can be dUT1R, dODR
    outputs:
        psd: power spectrum density
             unit in second
        freq: x-axis of psd
    """
    psd = np.abs(np.fft.rfft(x))
    psd = psd / len(psd)
    freq = np.arange(1, len(psd) + 1)
    return freq, psd


def powersp2(x):
    """
    inputs: 
        x: can be dUT1R, dODR
    outputs:
        psd: power spectrum density
    note:
        fre = freq+1
        fix the bug of x-axis alignment
        slower than powersp
    """
    n = len(x)
    # number of psd
    pnum = np.int(n/2)

    t = np.arange(1, n+1)

    # outputs init
    freq = np.arange(1, pnum+1)
    psd = []

    for i in freq:
        omega = 2.0 * np.pi * i * t / n
        spcos = x * np.cos(omega)
        spsin = x * np.sin(omega)
        si = np.sqrt((spcos.sum() * 2.0 / n) ** 2 + (spsin.sum() * 2.0 / n) ** 2)
        psd.append(si)
    freq = freq + 1
    return freq, psd


def powersp3(x):
    """
    inputs: 
        x: can be dUT1R, dODR

    outputs:
        psd: power spectrum density

    notes:
    use matrix multply, but slower than powersp2
    """
    # number of x
    n = len(x)
    # number of psd
    pnum = np.int(n / 2)

    t = np.arange(1, n)

    freq = np.arange(1, pnum+1)
    mat = np.zeros((pnum, n-1))
    mat[0:] = t
    for i in range(len(mat)):
        mat[i] = 2.0 * np.pi * mat[i] * (i + 1) / n
    spcos = np.cos(mat) * x[0:-1]
    spsin = np.sin(mat) * x[0:-1]
    psd = np.sqrt((spcos.sum(axis=1) * 2.0 / n) ** 2 + (spsin.sum(axis=1) * 2.0 / n) ** 2)
    return freq, psd


def func_ut1(arg, t):
    """
    inputs:
        arg: arguments that need to be determined by lsq
        t: np.ndarray, mjd in unit of day
    outputs:
          result: np.ndaray, ut1 model seires
    """
    a = arg[0]
    b = arg[1]
    c = arg[2]
    d = arg[3]
    e = arg[4]
    f = arg[5]
    g = arg[6]

    p1 = np.float64(365.240)
    p2 = np.float64(182.620)
    result = (a + b * t + c * t * t
              + d * np.cos(2.0 * np.pi * t / p1)
              + e * np.sin(2.0 * np.pi * t / p1)
              + f * np.cos(2.0 * np.pi * t / p2)
              + g * np.sin(2.0 * np.pi * t / p2))
    return result


def _func_ut1_res(arg, t, y):
    """
    indata:
        t: mjd in unit of day
        y: dUT1R or LODR
        arg: arguments that need to be determined by lsq
    outputs:
        residual: residuals of data-model
    """
    a = arg[0]
    b = arg[1]
    c = arg[2]
    d = arg[3]
    e = arg[4]
    f = arg[5]
    g = arg[6]

    p1 = np.float64(365.240)
    p2 = np.float64(182.620)
    residual = a + b * t + c * t * t + \
        d * np.cos(2.0 * np.pi * t / p1) + \
        e * np.sin(2.0 * np.pi * t / p1) + \
        f * np.cos(2.0 * np.pi * t / p2) + \
        g * np.sin(2.0 * np.pi * t / p2) - y
    return residual


def lsq_func_ut1(y, mjd):
    """
       inputs: 
           y: can be dUT1R, dLODR
           mjd
       outputs: 
           res_lsq[0]: best fitted argument [A,B,C,D,E,F,G]
           xfit: best fitted series
    """
    p0 = np.array([0, 0, 0, 0, 0, 0, 0])
    res_lsq = optimize.leastsq(_func_ut1_res, p0, args=(mjd, y))
    xfit = func_ut1(res_lsq[0], mjd)
    return res_lsq[0], xfit

def ar_forecast(series, p):
    """ forecast with ARMA methods """
    ARmodel=sm.tsa.ARMA(endog=series,order = (p,0,0)).fit(disp=-1, trend='nc', method='css')
    fcst=ARmodel.forecast()
    aic = ARmodel.aic
    bic = ARmodel.bic
    hqic = ARmodel.hqic
#   print("DW test, DW = %f"%sm.stats.durbin_watson(ARmodel.resid.values))
    print("p = %3i, aic = %f, bic = %f, hqic = %f, forecast ddUT1[+1] = %f"%(p, aic, bic, hqic ,fcst[0]))

    return aic, bic, hqic, fcst[0]

def next_day(inday):
    """return the datetime of next day
        inputs: "2018-01-01"
        outputs:"2018-01-02"
    """
    two_day = pd.date_range(start=inday, periods=2,freq='D')
    return two_day[1]

def next_n_days(inday,ndays):
    """return the datetime of next day
        inputs: "2018-01-01"
        outputs:"2018-01-02"
    """
    ndays = pd.date_range(start=inday, periods=ndays+1,freq='D')
    return ndays[1:]


def ARforecast_ut1_1(sut1,mjd,ic,max_lag):
    ################################################
    """forecast_ut1 of next day
       inputs: 
       sut1
       mjd
    diff 1 times
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
        

def ARforecast_ut1_2(sut1,mjd,ic,max_lag):
    ################################################
    """forecast_ut1 of next day
       inputs: 
       sut1
       mjd
       ic: aic, bic, hqic or t-stat
    diff 2 times
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


def _ARforecast_ut1_3(sut1,p,mjd):
    ################################################
    """forecast_ut1 of next day
       inputs: 
       sut1
       mjd
    diff 2 times
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

def read_eopc04(infile):
    raw_dta = pd.read_csv(infile, \
        skiprows=14, header=None, names=["year", "month", "day", "mjd", "pmx", \
        "pmy", "ut1", "lod", "dx", "dy", "epmx", "epmy", "eut1", \
        "elod","edx,", "edy"], delim_whitespace=True)
    
    start_datetime = "%4i-%02i-%02i"%(raw_dta.year[0],\
                                      raw_dta.month[0],raw_dta.day[0])
    
    end_datetime = "%4i-%02i-%02i"%(raw_dta.year[raw_dta.year.size-1],\
                                    raw_dta.month[raw_dta.month.size-1],\
                                    raw_dta.day[raw_dta.day.size-1])
    
    
    indx = pd.date_range(start=start_datetime,end=end_datetime,freq='D')
    raw_dta.index=indx
    return raw_dta

def read_usno(infile):
    raw_dta = pd.read_csv(infile, \
        skiprows=1, header=None, names=["jd", "pmx", \
        "pmy", "sut1", "epmx", "epmy", "eut1", \
        "coor_pmx_pmy","coor_pmx_ut1,", "coor_pmy_ut1"], delim_whitespace=True,comment='#')
    
    t = Time(raw_dta.jd[0],format='jd')
    start_datetime  = t.iso
    t = Time(raw_dta.jd[raw_dta.jd.size-1],format='jd')
    end_datetime  = t.iso
   
     
    mjd = Time(Time(raw_dta.jd,format='jd'),format='mjd').value

    indx = pd.date_range(start=start_datetime,end=end_datetime,freq='D')
    raw_dta.index=indx
    raw_dta.insert(0,'mjd',mjd)
    return raw_dta


def func_pmxy(arg, t):
    """
    inputs:
        arg: arguments that need to be determined by lsq
        t: np.ndarray, mjd in unit of day
    outputs:
          result: np.ndaray, ut1 model seires
    """
    a = arg[0]
    b = arg[1]
    d = arg[2]
    e = arg[3]
    f = arg[4]
    g = arg[5]

    p1 = np.float64(365.240)
    p2 = np.float64(435.000)
#   p2 = np.float64(432.080)
    result = (a + b * t  
              + d * np.cos(2.0 * np.pi * t / p1)
              + e * np.sin(2.0 * np.pi * t / p1)
              + f * np.cos(2.0 * np.pi * t / p2)
              + g * np.sin(2.0 * np.pi * t / p2))
    return result

def _func_pmxy_res(arg, t, y):
    """
    indata:
        t: mjd in unit of day
        y: can be pmx pmy
        arg: arguments that need to be determined by lsq
    outputs:
        residual: residuals of data-model
    """
    a = arg[0]
    b = arg[1]
    d = arg[2]
    e = arg[3]
    f = arg[4]
    g = arg[5]

    p1 = np.float64(365.240)
    p2 = np.float64(435.000)
#   p2 = np.float64(432.080)
    residual = a + b * t + \
        d * np.cos(2.0 * np.pi * t / p1) + \
        e * np.sin(2.0 * np.pi * t / p1) + \
        f * np.cos(2.0 * np.pi * t / p2) + \
        g * np.sin(2.0 * np.pi * t / p2) - y
    return residual


def lsq_func_pmxy(y, mjd):
    """
       inputs: 
           y: can be pmx, pmy
           mjd
       outputs: 
           res_lsq[0]: best fitted argument [A,B,C,D,E,F,G]
           xfit: best fitted series
    """
    p0 = np.array([0, 0, 0, 0, 0, 0])
    res_lsq = optimize.leastsq(_func_pmxy_res, p0, args=(mjd, y))
    xfit = func_pmxy(res_lsq[0], mjd)
    return res_lsq[0], xfit

def ARforecast_pmxy_1(pmxy,mjd,ic,max_lag):
    ################################################
    """forecast polar motion pmx pmy of next day
       inputs: 
       pmxy
       mjd
    diff 1 times
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
        
        
def ARforecast_pmxy_2(pmxy,mjd,ic,max_lag,pred_num):
    ################################################
    """forecast polar motion pmx pmy of next pred_num days
       inputs: 
       pmxy
       mjd
    diff 1 times
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
        
def ARforecast_ut1_3(sut1,mjd,ic,max_lag, pred_num):
    ################################################
    """forecast_ut1 of next day
       inputs: 
       sut1
       mjd
       ic: aic, bic, hqic or t-stat
    diff 2 times
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


def read_usno_hdr(infile):
    f=open(infile,'r')
    lines=f.readlines()
    f.close()
    for i in np.arange(len(lines)):
        if "Last date with real data" in lines[i]:
           end_day = lines[i].split()[-1].replace('.','-')
           pred_day = lines[i+1].split()[-1].replace('.','-')
           pred_num = (Time(pred_day)-Time(end_day)).value
    return end_day, pred_num

         
def ARforecast_pmxy_3(pmxy,mjd,ic,max_lag,pred_num):
    ################################################
    """forecast_ut1 of next day
       inputs: 
       pmxy
       mjd
    diff 1 times
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
 
def get_leap_second(mjd): 
     leap_sec_epochs = [57754, 57204,56109,54832,53736,51179,50630,50083,49534,49169,48804,48257, \
     47892,47161,46247,45516,45151,44786,44239,43874,43509, \
     43144,42778,42413,42048,41683,41499,41317] 
  
     sec = [37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,\
            16,15,14,13,12,11,10] 
  
     for i in range(len(leap_sec_epochs)): 
            if mjd >= leap_sec_epochs[i]: 
                   return sec[i] 
                   break


def func_dXdY(arg, t):
    """
    inputs:
        arg: arguments that need to be determined by lsq
        t: np.ndarray, mjd in unit of day
    outputs:
          result: np.ndaray, ut1 model seires
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
    result = (a + b * t + c * t * t
              + d * np.cos(2.0 * np.pi * t / p1)
              + e * np.sin(2.0 * np.pi * t / p1)
              + f * np.cos(2.0 * np.pi * t / p2)
              + g * np.sin(2.0 * np.pi * t / p2))
    return result

def _func_dXdY_res(arg, t, y):
    """
    indata:
        t: mjd in unit of day
        y: dUT1R or LODR
        arg: arguments that need to be determined by lsq
    outputs:
        residual: residuals of data-model
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

def lsq_func_dXdY(y, mjd,p0):
    """
       inputs: 
           y: can be dUT1R, dLODR
           mjd
       outputs: 
           res_lsq[0]: best fitted argument [A,B,C,D,E,F,G]
           xfit: best fitted series
    """
#   p0 = np.array([0, 0, 0, 0, 0, 0, 0,365.25,433])
    res_lsq = optimize.leastsq(_func_dXdY_res, p0, args=(mjd, y))
    xfit = func_dXdY(res_lsq[0], mjd)
    return res_lsq[0], xfit

def func_dPsidEps(arg, t):
    """
    inputs:
        arg: arguments that need to be determined by lsq
        t: np.ndarray, mjd in unit of day
    outputs:
          result: np.ndaray, ut1 model seires
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
    result = (a + b * t + c * t * t
              + d * np.cos(2.0 * np.pi * t / p1)
              + e * np.sin(2.0 * np.pi * t / p1)
              + f * np.cos(2.0 * np.pi * t / p2)
              + g * np.sin(2.0 * np.pi * t / p2)
              + m * np.cos(2.0 * np.pi * t / p3)
              + n * np.sin(2.0 * np.pi * t / p3))
    return result

def _func_dPsidEps_res(arg, t, y):
    """
    indata:
        t: mjd in unit of day
        y: dUT1R or LODR
        arg: arguments that need to be determined by lsq
    outputs:
        residual: residuals of data-model
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

def lsq_func_dPsidEps(y, mjd,p0):
    """
       inputs: 
           y: can be dUT1R, dLODR
           mjd
       outputs: 
           res_lsq[0]: best fitted argument [A,B,C,D,E,F,G]
           xfit: best fitted series
    """
    res_lsq = optimize.leastsq(_func_dPsidEps_res, p0, args=(mjd, y))
    xfit = func_dPsidEps(res_lsq[0], mjd)
    return res_lsq[0], xfit
        
def ARforecast_dXdY(dX,mjd,ic,max_lag,pred_num,p0):
    ################################################
    """forecast dX and dY
       inputs: 
       dX or dY
       mjd
    diff 1 times
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
 
