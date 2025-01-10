def func_ut1(arg, t):
    """Calculates UT1 model series based on input parameters.

    Args:
        arg (list): Coefficients determined by least squares fitting
        t (np.ndarray): Modified Julian Date in days

    Returns:
        np.ndarray: UT1 model series
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

def get_time():
    """Gets current system time in formatted string.

    Returns:
        str: Time string in format 'hh:mm:ss on YYYY/MM/DD'
    """

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

def jd2mjd(jd):
    """Converts Julian Date to Modified Julian Date.

    Args:
        jd (float): Julian Date

    Returns:
        float: Modified Julian Date
    """
    return jd - 2400000.5

def lsq_func_dPsidEps(y, mjd,p0):
    """Performs least squares fitting for dPsi/dEps parameters.

    Args:
        y (np.ndarray): Input data (dUT1R or dLODR)
        mjd (np.ndarray): Modified Julian Dates
        p0 (np.ndarray): Initial parameter estimates

    Returns:
        tuple: 
            np.ndarray: Best fit parameters [A,B,C,D,E,F,G]
            np.ndarray: Fitted series
    """
    res_lsq = optimize.leastsq(_func_dPsidEps_res, p0, args=(mjd, y))
    xfit = func_dPsidEps(res_lsq[0], mjd)
    return res_lsq[0], xfit

def lsq_func_dXdY(y, mjd,p0):
    """Performs least squares fitting for dX/dY parameters.

    Args:
        y (np.ndarray): Input data (dUT1R or dLODR)
        mjd (np.ndarray): Modified Julian Dates
        p0 (np.ndarray): Initial parameter estimates

    Returns:
        tuple:
            np.ndarray: Best fit parameters [A,B,C,D,E,F,G]
            np.ndarray: Fitted series
    """
#   p0 = np.array([0, 0, 0, 0, 0, 0, 0,365.25,433])
    res_lsq = optimize.leastsq(_func_dXdY_res, p0, args=(mjd, y))
    xfit = func_dXdY(res_lsq[0], mjd)
    return res_lsq[0], xfit

def lsq_func_pmxy(y, mjd):
    """Performs least squares fitting for polar motion (pmx/pmy).

    Args:
        y (np.ndarray): Input data (pmx or pmy)
        mjd (np.ndarray): Modified Julian Dates

    Returns:
        tuple:
            np.ndarray: Best fit parameters [A,B,C,D,E,F,G]
            np.ndarray: Fitted series
    """
    p0 = np.array([0, 0, 0, 0, 0, 0])
    res_lsq = optimize.leastsq(_func_pmxy_res, p0, args=(mjd, y))
    xfit = func_pmxy(res_lsq[0], mjd)
    return res_lsq[0], xfit

def lsq_func_ut1(y, mjd):
    """Performs least squares fitting for UT1 parameters.

    Args:
        y (np.ndarray): Input data (dUT1R or dLODR)
        mjd (np.ndarray): Modified Julian Dates

    Returns:
        tuple:
            np.ndarray: Best fit parameters [A,B,C,D,E,F,G]
            np.ndarray: Fitted series
    """
    p0 = np.array([0, 0, 0, 0, 0, 0, 0])
    res_lsq = optimize.leastsq(_func_ut1_res, p0, args=(mjd, y))
    xfit = func_ut1(res_lsq[0], mjd)
    return res_lsq[0], xfit

def mjd2jc(mjd):
    """Converts Modified Julian Date to Julian Century.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: Julian Century
    """
    mjd = (np.float64(mjd) - 51544.5) / 36525
    return mjd

def mjd2jd(mjd):
    """Converts Modified Julian Date to Julian Date.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: Julian Date
    """
    return mjd + 2400000.5
