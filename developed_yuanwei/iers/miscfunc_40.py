import os
import numpy as np
import pandas as pd
from astropy.time import Time

def next_day(inday):
    """Returns the datetime of the next day.

    Args:
        inday (str): Input date string in format "YYYY-MM-DD"

    Returns:
        pd.Timestamp: Datetime object for the next day

    Raises:
        ValueError: If input date format is invalid
    """
    try:
        pd.to_datetime(inday, format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Input date must be in format 'YYYY-MM-DD'")
    
    two_day = pd.date_range(start=inday, periods=2,freq='D')
    return two_day[1]


def next_n_days(inday, ndays):
    """Returns datetime objects for the next n days.

    Args:
        inday (str): Input date string in format "YYYY-MM-DD"
        ndays (int): Number of days to generate

    Returns:
        pd.DatetimeIndex: Array of datetime objects for the next n days

    Raises:
        ValueError: If input date format is invalid or ndays is negative
        TypeError: If ndays is not an integer
    """
    try:
        pd.to_datetime(inday, format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Input date must be in format 'YYYY-MM-DD'")
    
    if not isinstance(ndays, int):
        raise TypeError("ndays must be an integer")
    if ndays < 0:
        raise ValueError("ndays cannot be negative")
        
    ndays = pd.date_range(start=inday, periods=ndays+1,freq='D')
    return ndays[1:]


def powersp(x):
    """Calculates power spectrum density using FFT.

    Args:
        x (np.ndarray): Input time series data (e.g. dUT1R, dODR)

    Returns:
        tuple: 
            freq (np.ndarray): Frequency axis values
            psd (np.ndarray): Power spectrum density values in seconds

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If input array is empty
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")
        
    psd = np.abs(np.fft.rfft(x))
    psd = psd / len(psd)
    freq = np.arange(1, len(psd) + 1)
    return freq, psd

def powersp2(x):
    """Calculates power spectrum density using direct Fourier transform.

    Args:
        x (np.ndarray): Input time series data (e.g. dUT1R, dODR)

    Returns:
        tuple:
            freq (np.ndarray): Frequency axis values (shifted by +1)
            psd (np.ndarray): Power spectrum density values

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If input array is empty

    Note:
        This implementation fixes x-axis alignment but is slower than powersp()
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")

    n = len(x)
    pnum = np.int(n/2)
    t = np.arange(1, n+1)
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
    """Calculates power spectrum density using matrix multiplication.

    Args:
        x (np.ndarray): Input time series data (e.g. dUT1R, dODR)

    Returns:
        tuple:
            freq (np.ndarray): Frequency axis values
            psd (np.ndarray): Power spectrum density values

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If input array is empty

    Note:
        This implementation uses matrix multiplication but is slower than powersp2()
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(x) == 0:
        raise ValueError("Input array cannot be empty")

    n = len(x)
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


def read_eopc04(infile):
    """Reads EOP C04 format data file.

    Args:
        infile (str): Path to input file

    Returns:
        pd.DataFrame: DataFrame containing EOP data with datetime index

    Raises:
        FileNotFoundError: If specified file does not exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"File not found: {infile}")
    if not os.path.isfile(infile):
        raise ValueError(f"Path is not a file: {infile}")
        
    try:
        raw_dta = pd.read_csv(infile, \
            skiprows=14, header=None, names=["year", "month", "day", "mjd", "pmx", \
            "pmy", "ut1", "lod", "dx", "dy", "epmx", "epmy", "eut1", \
            "elod","edx,", "edy"], delim_whitespace=True)
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty or contains no data")
    except pd.errors.ParserError:
        raise ValueError("File format is invalid")
    
    start_datetime = "%4i-%02i-%02i"%(raw_dta.year[0],\
                                      raw_dta.month[0],raw_dta.day[0])
    
    end_datetime = "%4i-%02i-%02i"%(raw_dta.year[raw_dta.year.size-1],\
                                    raw_dta.month[raw_dta.month.size-1],\
                                    raw_dta.day[raw_dta.day.size-1])
    
    indx = pd.date_range(start=start_datetime,end=end_datetime,freq='D')
    raw_dta.index=indx
    return raw_dta


def read_usno(infile):
    """Reads USNO format data file.

    Args:
        infile (str): Path to input file

    Returns:
        pd.DataFrame: DataFrame containing USNO data with datetime index

    Raises:
        FileNotFoundError: If specified file does not exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"File not found: {infile}")
    if not os.path.isfile(infile):
        raise ValueError(f"Path is not a file: {infile}")
        
    try:
        raw_dta = pd.read_csv(infile, \
            skiprows=1, header=None, names=["jd", "pmx", \
            "pmy", "sut1", "epmx", "epmy", "eut1", \
            "coor_pmx_pmy","coor_pmx_ut1,", "coor_pmy_ut1"], delim_whitespace=True,comment='#')
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty or contains no data")
    except pd.errors.ParserError:
        raise ValueError("File format is invalid")
    
    t = Time(raw_dta.jd[0],format='jd')
    start_datetime  = t.iso
    t = Time(raw_dta.jd[raw_dta.jd.size-1],format='jd')
    end_datetime  = t.iso
    
    mjd = Time(Time(raw_dta.jd,format='jd'),format='mjd').value
    indx = pd.date_range(start=start_datetime,end=end_datetime,freq='D')
    raw_dta.index=indx
    raw_dta.insert(0,'mjd',mjd)
    return raw_dta


def read_usno_hdr(infile):
    """Reads header information from USNO format file.

    Args:
        infile (str): Path to input file

    Returns:
        tuple:
            end_day (str): Last date with real data in format "YYYY-MM-DD"
            pred_num (int): Number of predicted days

    Raises:
        FileNotFoundError: If specified file does not exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"File not found: {infile}")
    if not os.path.isfile(infile):
        raise ValueError(f"Path is not a file: {infile}")
        
    try:
        f=open(infile,'r')
        lines=f.readlines()
        f.close()
        for i in np.arange(len(lines)):
            if "Last date with real data" in lines[i]:
               end_day = lines[i].split()[-1].replace('.','-')
               pred_day = lines[i+1].split()[-1].replace('.','-')
               pred_num = (Time(pred_day)-Time(end_day)).value
        return end_day, pred_num
    except Exception as e:
        raise ValueError(f"Error reading file header: {str(e)}")


def sut1utc(ut1_utc):
    """Converts discontinuous UT1-UTC series to continuous series.

    Args:
        ut1_utc (Union[list, np.ndarray]): Input UT1-UTC series in seconds

    Returns:
        np.ndarray: Continuous UT1-UTC series with leap seconds accounted for

    Raises:
        TypeError: If input is not a list or numpy array
        ValueError: If input array is empty
    """
    if not isinstance(ut1_utc, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array")
    if len(ut1_utc) == 0:
        raise ValueError("Input array cannot be empty")
        
    ut1_utc = np.array(ut1_utc)
    sut1_utc = ut1_utc
    dut1_utc = ut1_utc[1:] - ut1_utc[0:-1]
    index = np.argwhere(dut1_utc > 0.9)
    index = np.reshape(index, len(index))
    for i in index:
        sut1_utc[i+1:] = sut1_utc[i+1:] - 1.0
    return sut1_utc
