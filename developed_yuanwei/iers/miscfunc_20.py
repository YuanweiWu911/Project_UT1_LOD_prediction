def _func_pmxy_res(arg, t, y):
    """Calculates residuals between polar motion data and model.

    Args:
        t (np.ndarray): Modified Julian Date in days
        y (np.ndarray): Polar motion values (pmx or pmy)
        arg (list): Model coefficients to be determined by least squares

    Returns:
        np.ndarray: Residuals between input data and model
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

def _func_ut1_res(arg, t, y):
    """Calculates residuals between UT1/LOD data and model.

    Args:
        t (np.ndarray): Modified Julian Date in days
        y (np.ndarray): UT1 or LOD values
        arg (list): Model coefficients to be determined by least squares

    Returns:
        np.ndarray: Residuals between input data and model
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

def acumu(dy, y0, order=1):
    """Accumulates difference results to recover original values.

    Args:
        dy (np.ndarray): Difference results to accumulate
        y0 (float): Initial value
        order (int, optional): Accumulation direction. 
            1 for forward accumulation, -1 for backward accumulation. Defaults to 1.

    Returns:
        np.ndarray: Accumulated values matching original input dimensions
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

def ar_forecast(series, p):
    """Forecasts time series using ARMA methods.

    Args:
        series (pd.Series): Time series data to forecast
        p (int): Order of autoregressive component

    Returns:
        tuple: Contains:
            - aic (float): Akaike Information Criterion
            - bic (float): Bayesian Information Criterion  
            - hqic (float): Hannan-Quinn Information Criterion
            - forecast (float): Predicted value for next time step
    """
    ARmodel=sm.tsa.ARMA(endog=series,order = (p,0,0)).fit(disp=-1, trend='nc', method='css')
    fcst=ARmodel.forecast()
    aic = ARmodel.aic
    bic = ARmodel.bic
    hqic = ARmodel.hqic
#   print("DW test, DW = %f"%sm.stats.durbin_watson(ARmodel.resid.values))
    print("p = %3i, aic = %f, bic = %f, hqic = %f, forecast ddUT1[+1] = %f"%(p, aic, bic, hqic ,fcst[0]))

    return aic, bic, hqic, fcst[0]

def d2jd(tstr):
    """Converts date string to Julian Date, Modified Julian Date, and year-day format.

    Args:
        tstr (str): Date string in 'YYYY-MM-DD' format

    Returns:
        tuple: Contains:
            - jd (float): Julian Date
            - mjd (float): Modified Julian Date
            - year_day (int): Combined year and day of year (YYYYDDD)

    Note:
        toordinal() returns days since December 31, 1 BC at 00:00
        Julian day counts days since January 1, 4713 BC at 12:00
        The conversion offset is 1721424.5 days
    """
    fmt = '%Y-%m-%d'
    dt = datetime.strptime(tstr, fmt)
    tt = dt.timetuple()
    jd = dt.toordinal()+1721424.5
    mjd = jd - 2400000.5
    return jd, mjd, tt.tm_year * 1000 + tt.tm_yday

def diff(y, order=1):
    """Calculates finite differences of an array using specified method.

    Args:
        y (Union[list, np.ndarray]): Input array to differentiate
        order (int, optional): Difference method:
            -1: backward difference
             1: forward difference
             0: central difference
            Defaults to 1.

    Returns:
        np.ndarray: Array of differences with length reduced by 1
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

def extrap(y, order):
    """Extrapolates array values at both ends using interpolation.

    Args:
        y (Union[list, np.ndarray]): Input array to extrapolate
        order (int): Interpolation order:
            1: linear
            2: 2nd order polynomial
            3: cubic

    Returns:
        np.ndarray: Input array with extrapolated values added at both ends
    """
    leny = len(y)
    x = range(0, leny)
    f = InterpolatedUnivariateSpline(x, y, k=order)
    result = np.insert(y, 0, f([-1]))
    result = np.append(result, f([leny]))
    return result

def func_dPsidEps(arg, t):
    """Calculates UT1 model series using periodic components.

    Args:
        arg (list): Model coefficients to be determined by least squares:
            [a, b, c, d, e, f, g, m, n, t1, t2, t3]
        t (np.ndarray): Modified Julian Date in days

    Returns:
        np.ndarray: UT1 model series combining linear, quadratic and periodic terms
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

def func_dXdY(arg, t):
    """Calculates UT1 model series using periodic components.

    Args:
        arg (list): Model coefficients to be determined by least squares:
            [a, b, c, d, e, f, g, t1, t2]
        t (np.ndarray): Modified Julian Date in days

    Returns:
        np.ndarray: UT1 model series combining linear, quadratic and periodic terms
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

def func_pmxy(arg, t):
    """Calculates polar motion model series using periodic components.

    Args:
        arg (list): Model coefficients to be determined by least squares:
            [a, b, d, e, f, g]
        t (np.ndarray): Modified Julian Date in days

    Returns:
        np.ndarray: Polar motion model series combining linear and periodic terms
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
