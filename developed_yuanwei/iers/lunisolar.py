# lunisolar.py
#
# The expressions for the fundamental arguments of nutation are given by 
# IERS convention 2010, Section 5.7.2
#
# writer: yuanwei Wu @NTSC 2019-08-14
# version 1.0
# 2019-08-14
# 2019-08-15: np.mod different from np.fmod, DUT1 are consistent with IERS fortran
#             code at accuracy of 1E-14 second
# 2019-08-16 calc_dut1 tested
#
import pkg_resources
import numpy as np
from numpy.polynomial.polynomial import polyval as polyval


def fundarg(mjd):
    """ fundarg is copied from FUNDARG.F
     used to test lunisolarf5
     """
    # Arcseconds to radians
    # DOUBLE PRECISION das2r
    das2r = np.float64(4.848136811095359935899141E-6)

    #  Arcseconds in a full circle
    #     DOUBLE PRECISION turnas
    turnas = np.float64(1296000E0)

    # 2Pi
    # DOUBLE PRECISION D2PI
    # D2PI = np.float64(6.283185307179586476925287E0)

    t = (np.float64(mjd) - 51544.5) / 36525
    #
    #  Compute the fundamental argument L.
    el = np.fmod(np.float64(485868.249036E0) +
                 t * (np.float64(1717915923.2178E0) +
                      t * (np.float64(31.8792E0) +
                           t * (np.float64(0.051635E0) +
                                t * np.float64(-0.00024470E0)))), turnas) * das2r

    # Compute the fundamental argument LP.
    lp = np.fmod(1287104.79305E0 +
                 t * (129596581.0481E0 +
                      t * (-0.5532E0 +
                           t * (0.000136E0 +
                                t * -0.00001149E0))), turnas) * das2r

    # Compute the fundamental argument F.
    f = np.fmod(335779.526232E0 +
                t * (1739527262.8478E0 +
                     t * (-12.7512E0 +
                          t * (-0.001037E0 +
                               t * 0.00000417E0))), turnas) * das2r

    #  Compute the fundamental argument D.
    d = np.fmod(1072260.70369E0 +
                t * (1602961601.2090E0 +
                     t * (-6.3706E0 +
                          t * (0.006593E0 +
                               t * -0.00003169E0))), turnas) * das2r

    #  Compute the fundamental argument OM.
    om = np.fmod(np.float64(450160.3980360) +
                 t * (np.float64(-6962890.54310) +
                      t * (np.float64(7.47220) +
                           t * (np.float64(0.0077020) +
                                t * np.float64(-0.000059390)))), turnas) * das2r

    return el, lp, f, d, om


def lunisolarf5(mjd):
    """
    calculate lunisolar nuation parameters of 
    l, l', F, D, Omega (iers2010, sect 5.7.2)
    input: 
    mjd in unit of day
    output:     
    F1 to F5 in unit of degree
    F1: l = Mean Anomaly of the Moon
    F2: l'= Mean Anomaly of the Sun
    F3: F = L - omega
    F4: D = Mean Elongation of the Moon from the Sun
    F5: Omega = Mean Longitude of the Ascending Node of the Moon
    """
    # arg[0]   0 order in unit of degree
    # arg[2:5] 1st to 4th order in unit of arcsec
    table_path = 'data2010/lunisolar_nutation.txt'
    filepath = pkg_resources.resource_filename(__name__, table_path)
    dtype1 = np.dtype([('col1', 'float64'), ('col2', 'float64'),
                       ('col3', 'float64'), ('col4', 'float64'),
                       ('col5', 'float64'), ('col6', 'float64')])
    (col1, col2, col3, col4, col5, col6) = np.loadtxt(filepath, dtype=dtype1, delimiter='|', unpack=True)
    data = np.transpose(np.array([col1, col2, col3, col4, col5, col6]))
    arg1 = data[0]
    arg2 = data[1]
    arg3 = data[2]
    arg4 = data[3]
    arg5 = data[4]

    t = (np.float64(mjd) - 51544.5) / 36525

    f1 = np.fmod(np.deg2rad(arg1[0] + polyval(t, arg1[1:]) / 3600.0), 2.0 * np.pi)
    f2 = np.fmod(np.deg2rad(arg2[0] + polyval(t, arg2[1:]) / 3600.0), 2.0 * np.pi)
    f3 = np.fmod(np.deg2rad(arg3[0] + polyval(t, arg3[1:]) / 3600.0), 2.0 * np.pi)
    f4 = np.fmod(np.deg2rad(arg4[0] + polyval(t, arg4[1:]) / 3600.0), 2.0 * np.pi)
    f5 = np.fmod(np.deg2rad(arg5[0] + polyval(t, arg5[1:]) / 3600.0), 2.0 * np.pi)

    return np.array([f1, f2, f3, f4, f5])


def tidal_table():
    """
    read IERS2010 tables 8.1 from file
    return l,l2,FD,Omega,period,B0,C0,B1,C1,B2,C2

    find a inconsistent of 55th row of C1 for DLOD,
    IERS pdf table8.1 0.0263
    IERS fotran  code 0.0267
    with diff 0.0004
    """
    table_path = 'data2010/lunisolar_table0.txt'
    filepath = pkg_resources.resource_filename(__name__, table_path)
    dtype1 = np.dtype([('l', 'int'), ('l2', 'int'), ('F', 'int'), ('D', 'int'), ('Omega', 'int'),
                       ('period', 'float64'), ('B0', 'float64'), ('C0', 'float64'),
                       ('B1', 'float64'), ('C1', 'float64'), ('B2', 'float64'), ('C2', 'float64')])
    (l, l2, F, D, Omega, period, B0, C0, B1, C1, B2, C2) = np.loadtxt(filepath, dtype=dtype1, delimiter='|',
                                                                      unpack=True)

    return np.array([l, l2, F, D, Omega, period, B0, C0, B1, C1, B2, C2])


def calc_dut1(mjd):
    """
    input:
         mjd in unit of DAY

    output:
         DUT1 in unit of second
         DLOD in unit of second
         DOMEGA in unit of rad/s

    note:
         DUT1 accuracy consistent with VIEVS at 1E-6 second
         and  with IERS Fortran code at 1E-13 second
    """
    mjd_nutation5 = np.transpose(lunisolarf5(mjd))
    ##################################################
    table8_1 = tidal_table()
    # 5*62 matrix
    argument5_62 = table8_1[0:5]
    # array len=62
    # DUT1
    b0_62 = table8_1[6]
    c0_62 = table8_1[7]
    # DLOD
    b1_62 = table8_1[8]
    c1_62 = table8_1[9]
    # DOMEGA
    b2_62 = table8_1[10]
    c2_62 = table8_1[11]
    ################################################
    # i points * 62 elements
    dut1s = []
    dlods = []
    domes = []
    for i in range(len(mjd_nutation5)):
        mat_62_5 = mjd_nutation5[i] * np.transpose(argument5_62)
        mat_62 = mat_62_5.sum(axis=1)
        #
        dut1 = b0_62 * np.sin(mat_62) + c0_62 * np.cos(mat_62)
        dut1s.append(dut1.sum() * 1.0E-4)

        dlod = b1_62 * np.cos(mat_62) + c1_62 * np.sin(mat_62)
        dlods.append(dlod.sum() * 1.0E-5)

        domega = b2_62 * np.cos(mat_62) + c2_62 * np.sin(mat_62)
        domes.append(domega.sum() * 1.0E-14)
    return np.array(dut1s), np.array(dlods), np.array(domes)
