# -*-coding:utf-8-*-
# __main__.py

import sys


if __name__ == '__main__':

    import numpy as np

    import iers

    from matplotlib import pyplot as plt

    #
    print('---- Start time: {}'.format(iers.miscfunc.get_time()))
    # test iers.lunisolar.calc_dut1
    #
    #
    labels = ['1960', '1970', '1980', '1990', '2000', '2010', '2020']
    labels = ['1980', '1990', '2000', '2010', '2020']
    yr_xtickv = []
    for i in labels:
        
        a, vmjd, c = iers.d2jd(i+'-01-01')
        yr_xtickv.append(vmjd)
    
    run_flag = True
    if run_flag == True:
        fig=plt.figure()
        ax1=fig.add_subplot(311)
        ax2=fig.add_subplot(312)
        ax3=fig.add_subplot(313)
    
        a, b, c = iers.d2jd('1960-01-01')
        mjd = b+np.arange(0,365*50,1)  #1970-01-01
        T = (np.float64(mjd) - 51544.5) / 36525
        ###################################################
        a,b,c=iers.d2jd('1990-01-01')
        mjd = b+np.arange(0,365*20,1)  #1970-01-01
        ###################################################
        mjd=np.array([54465])
        #mjd=np.arange(44239,58504,1)
        mjd=np.arange(51544,58504,1)
        ##################################################
        #(DUT1,DLOD,DOMEGA) = iers.lunisolar.calc_dut1(mjd)
        T=(np.float64(mjd)-51544.5)/36525
        (DUT1,DLOD,DOMEGA) = iers.lunisolar.calc_dut1(mjd)
        if len(mjd)>1:
           ax1.plot(mjd,DUT1,'b+')
           ax2.plot(mjd,DLOD)
           #ax3.plot((mjd-51544.5)/365.25+2000,DOMEGA)
           
           MJD,DUT1_V = np.loadtxt('iers/data2010/tide1960-2020-vievs.txt',dtype='float64',unpack=True)
           ax1.plot(MJD,DUT1_V,'o',mec='r',mfc='none')
           ax3.plot(mjd,DUT1_V-DUT1,'+')
           
        else:
           print("T   =%+25.20e"%T)
           print("DUT1=%+25.20e"%DUT1)
           print("DLOD=%+25.20e"%DLOD)
           print("DOMG=%+25.20e"%DOMEGA)
    
    ###################################################
    #test iers.miscfunc
    run_flag = True
    if run_flag == True:
        fig=plt.figure()
        ax1=fig.add_subplot(511)
        ax2=fig.add_subplot(512)
        ax3=fig.add_subplot(513)
        ax4=fig.add_subplot(514)
        ax5=fig.add_subplot(515)
    
        a,mjd_start,c=iers.d2jd('1980-01-01')
        a,mjd_end,c=iers.d2jd('2018-12-31')
    
        mjd, UT1_UTC,LOD=np.loadtxt('/home/aips/EOP_DOWN/20190815/eopc04_IAU2000.62-now',usecols=[3,6,7],dtype='float',unpack=True,comments='*',skiprows=14)
    
        start_ind = np.where(mjd>=mjd_start)[0]
        end_ind = np.where(mjd<=mjd_end)[0]
    
        mjd=mjd[start_ind[0]:end_ind[-1]+1]
        UT1_UTC = UT1_UTC[start_ind[0]:end_ind[-1]+1]
        LOD = LOD[start_ind[0]:end_ind[-1]+1]
        ##########################
        #SUT1_UTC continous UT1-UTC series
        SUT1_UTC=iers.miscfunc.SUT1_UTC(UT1_UTC)
    
        #zonal tidal for UT1, LOD and Omega
        (DUT1,DLOD,DOMEGA) = iers.lunisolar.calc_dut1(mjd)
        LODR=LOD-DLOD
    
        #SUT1_UTC minus the zonal tidal effects
        SUT1_UTCR=SUT1_UTC-DUT1
    
        #1st order foward difference
        dSUT1_UTCR=iers.diff(SUT1_UTCR,1)
    
        #2nd order foward difference
        ddSUT1_UTCR=iers.diff(dSUT1_UTCR,1)
    
        #recover SUT1_UTCR from 1st order diff
        IntdSUT1_UTCR=iers.acumu(dSUT1_UTCR,SUT1_UTCR[0],1)
    
        #recover 1st order diff from 2nd order diff
        IntddSUT1_UTCR=iers.acumu(ddSUT1_UTCR,dSUT1_UTCR[0],1)
    
        #recover SUT1_UTCR from 2nd order diff
        IntIntddSUT1_UTCR=iers.acumu(IntddSUT1_UTCR,SUT1_UTCR[0],1)
    
        if len(mjd)>1:
            ax1.plot(mjd,SUT1_UTCR,'k-')
            ax2.plot(mjd,dSUT1_UTCR,'k.-')
            ax3.plot(mjd,ddSUT1_UTCR,'b.-')
    
        #test lsq_fit of the UT1R/LODR series
            lsq_res,x_fit = iers.miscfunc.lsq_func_UT1(ddSUT1_UTCR,mjd)
            print("least squre fit results: ",lsq_res)
            ax4.plot(mjd,x_fit,'k-')
            ax5.plot(mjd,ddSUT1_UTCR-x_fit,'k-')
        else:
           print("DUT1=%+25.20e"%DUT1)
           print("UT1_UTC=%+25.20e"%UT1_UTC)

        ax1.set_xlabel("year")
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        ax1.set_xticks(yr_xtickv)
        ax1.set_xticklabels(labels)
    ##################################################
    #test miscfunc.diff & acumu
    run_flag = True
    if run_flag == True:
       fig=plt.figure()
       ax1=fig.add_subplot(211)
       ax2=fig.add_subplot(212)
    
       y=np.sin(np.arange(-1,1.1,0.1)*np.pi)+3.0
       dy0=iers.diff(y,0)
       dy1=iers.diff(y,1)
       dy_1=iers.diff(y,-1)
    
       ax1.plot(y,'k-')
       ax2.plot(dy1,'ro-')
       ax2.plot(dy_1[1:],'bo')
    
       Intdy1=iers.acumu(dy1,y[0],1)
       Intdy_1=iers.acumu(dy_1,y[0],-1)
       ax1.plot(Intdy_1,'ro')
    
    ###################################################
    #test iers.miscfunc.powersp3
    run_flag = True
    if run_flag == True:
    
        fig2=plt.figure()
        f2_ax1=fig2.add_subplot(111)
    
        a,mjd_start,c=iers.d2jd('1961-01-01')
        a,mjd_end,c=iers.d2jd('2018-12-31')
        jc_start = iers.miscfunc.mjd2jc(mjd_start)
        jc_end = iers.miscfunc.mjd2jc(mjd_end)
        span_yr = (jc_end-jc_start)*100.0
    
        mjd, PMX,PMY,UT1_UTC,LOD=np.loadtxt('/home/aips/EOP_DOWN/20190815/eopc04_IAU2000.62-now',usecols=[3,4,5,6,7],dtype='float',unpack=True,comments='*',skiprows=14)
    
        start_ind = np.where(mjd>=mjd_start)[0]
        end_ind = np.where(mjd<=mjd_end)[0]
    
        mjd=mjd[start_ind[0]:end_ind[-1]+1]
        UT1_UTC = UT1_UTC[start_ind[0]:end_ind[-1]+1]
        LOD = LOD[start_ind[0]:end_ind[-1]+1]
        PMX = PMX[start_ind[0]:end_ind[-1]+1]
        PMY = PMY[start_ind[0]:end_ind[-1]+1]
    
        #LODR
        (DUT1,DLOD,DOMEGA) = iers.lunisolar.calc_dut1(mjd)
        LODR=LOD-DLOD
        dLODR=iers.diff(LODR,1)
    
        #SUT1_UTC
        SUT1_UTC=iers.miscfunc.SUT1_UTC(UT1_UTC)
        SUT1_UTCR=SUT1_UTC-DUT1
    
        #1st order foward difference
        dSUT1_UTCR=iers.diff(SUT1_UTCR,1)
    
        #2nd order foward difference
        ddSUT1_UTCR=iers.diff(dSUT1_UTCR,1)
    #########################################################
    #   print iers.miscfunc.get_time()
    #   power spectra of LOD and UT1
    #   freq,spd = iers.powersp(LODR)
    #   f2_ax1.plot(freq/span_yr,spd,'ro-')
    
        freq,spd = iers.powersp(dSUT1_UTCR)
        f2_ax1.plot(freq/span_yr,spd,'b+-')
    
    #   freq,spd = iers.powersp(PMX)
    #   f2_ax1.plot(freq/span_yr,spd,'ro-')
    
    #   f2_ax1.set_xlim([0.5,12])
    #########################################################
    print("---- End   time: %s"%iers.miscfunc.get_time())
    plt.show()
