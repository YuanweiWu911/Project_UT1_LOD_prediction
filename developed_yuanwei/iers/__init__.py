# __init__.py
"""
  iers
  =====

  Provides
    1. main purpose to predict UT1/LOD
       with LS+AR methods
    2. include functions to calculate lunar solar tidal effects for UT1/LOD/AAM
    3. difference of UT1/LOD
       accumulate of diff(UT1), diff(LOD)
    4. produce power spectra of UT1/LOD/PMX/PMY series
    5. least square fit of UT1/LOD series

  The py3_test_function.py 
"""
from .lunisolar import lunisolarf5, tidal_table, calc_dut1
from .miscfunc import d2jd, sut1utc, extrap, diff, acumu, powersp, powersp2, powersp3,\
     lsq_func_ut1, ar_forecast, func_ut1, next_day, next_n_days,\
     ARforecast_ut1_2, read_eopc04, ARforecast_ut1_1, ARforecast_ut1_3,\
     read_usno, ARforecast_pmxy_1,lsq_func_pmxy, func_pmxy, \
     ARforecast_pmxy_2, read_usno_hdr, get_leap_second, \
     lsq_func_dXdY,func_dXdY,lsq_func_dPsidEps,func_dPsidEps, \
     ARforecast_dXdY
     
__all__ = ['lunisolarf5', 'tidal_table', 'd2jd', 'sut1utc', 'extrap', \
          'diff', 'acumu', 'powersp', 'powersp2', 'powersp3', 'lsq_func_ut1','ar_forecast', \
          'calc_dut1','func_ut1','ARforecast_ut1_2','ARforecast_ut1_3',\
          'read_eopc04','next_day','ARforecast_ut1_1','next_n_days',\
          'read_usno','ARforecast_pmxy_1','lsq_func_pmxy','func_pmxy',\
          'ARforecast_pmxy_2','read_usno_hdr','get_leap_second',\
          'lsq_func_dXdY','func_dXdY','lsq_func_dPsidEps','func_dPsidEps',\
          'ARforecast_dXdY']

# __version__= get_version()
