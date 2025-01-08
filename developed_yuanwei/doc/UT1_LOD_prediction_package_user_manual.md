# UT1/LOD Prediction Software User Manual

## Overview
This Python package provides tools for predicting Universal Time 1 (UT1) and Length of Day (LOD) variations using LS+AR methods. It implements algorithms based on IERS conventions and includes functionality for:

- Lunar/solar tidal effect calculations
- UT1/LOD time series analysis
- Power spectrum analysis
- Least squares modeling
- ARMA-based forecasting
- Earth Orientation Parameter (EOP) data processing

## Package Structure
The package consists of three main modules:

1. **lunisolar.py** - Lunar/solar tidal calculations
2. **miscfunc.py** - Utility functions and forecasting
3. **__init__.py** - Package initialization and exports

## Key Features

### Lunar/Solar Tidal Effects
- `lunisolarf5()`: Calculates lunisolar nutation parameters
- `calc_dut1()`: Computes DUT1, DLOD, DOMEGA values
- `tidal_table()`: Reads IERS2010 tidal tables

### Time Series Analysis
- `powersp()`: Power spectrum density calculation
- `diff()`: Numerical differentiation
- `acumu()`: Numerical integration
- `extrap()`: Time series extrapolation

### Forecasting
- `ARforecast_ut1_1/2/3()`: UT1 forecasting methods
- `ARforecast_pmxy_1/2/3()`: Polar motion forecasting
- `ar_forecast()`: Generic ARMA forecasting

### Data Processing
- `read_eopc04()`: Reads EOP C04 format files
- `read_usno()`: Reads USNO format files
- `get_leap_second()`: Retrieves leap second information

## Usage Examples

### Basic Setup
```python
from iers import lunisolarf5, calc_dut1, ARforecast_ut1_1
```

### Lunar/Solar Calculations
```python
# Calculate lunisolar nutation parameters
mjd = 59000.0
f1, f2, f3, f4, f5 = lunisolarf5(mjd)

# Compute tidal effects
dut1, dlod, domega = calc_dut1(mjd)
```

### Time Series Analysis
```python
# Calculate power spectrum
freq, psd = powersp(ut1_series)

# Numerical differentiation
diff_series = diff(ut1_series)
```

### Forecasting
```python
# Forecast UT1 for next day
new_mjd, forecast_ut1, p = ARforecast_ut1_1(ut1_series, mjd_series, 'aic', 100)
```

### Data Processing
```python
# Read EOP C04 data
eop_data = read_eopc04('eopc04.1962-now')

# Get leap second information
leap_sec = get_leap_second(mjd)
```

## Input/Output Formats

### Input Data
- MJD (Modified Julian Date) as primary time format
- UT1-UTC series in seconds
- LOD series in seconds
- Polar motion coordinates in arcseconds

### Output Data
- Forecast results in same units as input
- Power spectra in frequency/amplitude format
- Tidal effects in seconds

## Dependencies
The package requires:
- NumPy
- Pandas
- SciPy
- statsmodels
- astropy

## References
- IERS Conventions (2010)
- USNO Earth Orientation Products
- IERS EOP C04 Series
