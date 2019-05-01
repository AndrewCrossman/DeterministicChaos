# -*- coding: utf-8 -*-
"""
@Title:     Physics 660 Project One
@Author     Andrew Crossman
@Date       Mar. 2nd, 2019
"""
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Helper function for calculating the Lyapunov fit
def func(t, b):
     return np.exp(b*t)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Updates the x-positions and velocities according to the Kinematic equations of Motion
def update(x1old, v1old, x2old, v2old, dt):
    v1new = v1old - dt
    v2new = v2old - dt
    x1new = x1old + v1old*dt - .5*dt**2
    x2new = x2old + v2old*dt - .5*dt**2
    return x1new, v1new, x2new, v2new
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Simulates elastic collision and returns the corresponding velocities    
def collision(m1,v1,m2,v2):
    newv1 = ((m1-m2)*v1 + ((2*m2)*v2))/(m1+m2)
    newv2 = ((2*m1)*v1 + ((m2-m1)*v2))/(m1+m2)
    return newv1,newv2  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Finds the exact time it takes for ball m1 to hit the ground 
def timetofall(x,v):
    t = v + ((v**2)+2*x)**(1/2)
    return t
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Finds the exact time it takes for ball m1 to collide with ball m2 from the floor 
def timetocollide(x1,v1,x2,v2):
    t = (x2-x1)/(v1-v2)
    return t
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
#Runs simulation that plots Poincar'e Sections, Positions, and Auto-Correlations
def chaos_balls(m1,m2,x1i,v1i,x2i,v2i):
    #Calculate Initial Energy
    m = m1 + m2
    E = (1/2)*(m1*v1i**2 + m2*v2i**2) + m1*9.807*x1i + m2*9.807*x2i
    print(E)
    g = 9.807
    x_scale = E/(m*g)
    v_scale = (E/m)**(1/2)
    t_scale = (E/(m*g**2))**(1/2)
    Eini = 0.5*(m1*(v1i/v_scale)**2 + m2*(v2i/v_scale)**2)/m + (m1*(x1i/x_scale) + m2*(x2i/x_scale))/m
    print ('Initial Energy = ',Eini)
    if abs(Eini - 1.0) > 1.e-6:
        print ('\nInitial Energy not equal to 1. Stopping execution.\n')
        return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Find data for Poincar'e Sections
    x2ps = []
    v2ps = []
    x1,v1,x2,v2 = x1i/x_scale,v1i/v_scale,x2i/x_scale,v2i/v_scale
    i = 1000 #number of collisions
    ttofall = timetofall(x1,v1)
    ttocollide  = float
    time = 0
    while i > 0:
        isbouncing=True
        while isbouncing:
            #Balls fall, update points accordingly
            ttofall = timetofall(x1,v1)
            time = time + ttofall
            x1,v1,x2,v2 = update(x1,v1,x2,v2,ttofall)
            x1, v1 = 0, -1*v1
            #Where would the balls be if they collide. Is it below x=0? If so bounce again
            ttocollide = timetocollide(x1,v1,x2,v2)
            x1t,v1t,x2t,v2t = update(x1,v1,x2,v2,ttocollide)
            #Does the ball bounce again?
            if x1t<0:
                ttofall = timetofall(x1,v1)
            else:
                isbouncing = False
        #Balls collide, update points accordingly        
        time = time + ttocollide
        x1,v1,x2,v2 = update(x1,v1,x2,v2,ttocollide)
        v1,v2 = collision(m1,v1,m2,v2)
        x2ps.append(x2)
        v2ps.append(v2)
        i-=1
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Find data for x1,v1,x2,v2 positions at equal time intervals across time it took to find 10000 collisions
    steps = 100000
    x_1 = np.zeros(steps)
    x_2 = np.zeros(steps)
    v_1 = np.zeros(steps)
    v_2 = np.zeros(steps)
    x_1_total = 0
    x_2_total = 0
    times = np.linspace(0,time,steps,endpoint=True)
    x_1[0],v_1[0],x_2[0],v_2[0] = x1i/x_scale,v1i/v_scale,x2i/x_scale,v2i/v_scale
    count = 0
    while count<len(times)-1:
        count+=1
        #Adjust current values for collisions and bounces if they occured between time steps
        event=True
        x1,v1,x2,v2 = x_1[count-1],v_1[count-1],x_2[count-1],v_2[count-1]
        while event:
            #print(count,x1,v1,x2,v2)
            timepassed=0
            #Previous timesteps values, used to accurately find current values
            #Does ball one hit the floor
            if x1<=0:
                #When did ball one hit the floor between time[count] and time[count-1]
                ttofall = timetofall(x1,v1)
                #Update values across delta-time = ttofall
                x1,v1,x2,v2 = update(x1,v1,x2,v2,ttofall)
                x1,v1 = 0, -1*v1
                #Increased timepassed
                timepassed+=ttofall
                #Update values at 
                x1,v1,x2,v2 = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
            #Have the balls collided    
            elif x1>x2:
                #When did the balls collide?
                ttocollide = timetocollide(x1,v1,x2,v2)
                #update values at across delta-time = ttocollide
                x1,v1,x2,v2 = update(x1,v1,x2,v2,ttocollide)
                v1,v2 = collision(m1,v1,m2,v2)
                #Increase timepassed
                timepassed+=ttocollide
                #Update values at 
                x1,v1,x2,v2 = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
            #If the balls are finished colliding/bouncing then update values correctly and leave loop
            else:
                #Update accurate values
                x_1[count],v_1[count],x_2[count],v_2[count] = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
                x_1_total = x_1_total + x_1[count]
                x_2_total = x_2_total + x_2[count]
                event=False
    print(times[-1])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Find data for auto-correlation functions
    print("correlating...")
    autocorrelation1 = np.zeros(int(steps/5))
    autocorrelation2 = np.zeros(int(steps/5))
    lag = []
    t = 0
    x_1_mean = x_1_total/len(x_1)
    x_2_mean = x_2_total/len(x_2)
    while t < len(autocorrelation1):
        suma, sumb = 0,0
        i = 0
        while i + t < steps:
            x1a = x_1[i] - x_1_mean
            x1b = x_1[i+t] - x_1_mean
            x2a = x_2[i] - x_2_mean
            x2b = x_2[i+t] - x_2_mean
            suma = suma + x1a * x1b
            sumb = sumb + x2a * x2b
            i+=1
        autocorrelation1[t] = suma
        autocorrelation2[t] = sumb
        lag.append(t*(times[1]-times[0]))
        #print(t,steps/25)
        t+=1
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Create all Graphs
    #Plot Poincar'e Sections    
    #NOTE: Poincare plots dont work with 10000 colllisons when set to plot, works fine for scatter though
    f,ax = plt.subplots()
    f.tight_layout()
    ax.scatter(x2ps,v2ps,alpha=.5,color='g',label="Mass Two")
    ax.set_title("Poincar´e Sections: "+r'$m_1 = $'+str(m1)+" "+r'$v_1 = $'+str(v1i)+" "+r'$m_2 = $'+str(m2)+" "+r'$v_2 = $'+str(v2i),style='italic')
    ax.set_ylabel("Velocity of "+r'$Mass_2 $'+r'$\left(\frac{v_2E^{1/2}}{m^{1/2}}\right)$',style='italic')
    ax.set_xlabel("Position of "+r'$Mass_2 $'+r'$\left(\frac{x_2E}{mg}\right)$',style='italic')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.legend(loc='upper right')
    f.show()
    f.savefig("Section-MassOne"+str(m1)+"MassTwo"+str(m2)+"x1i"+str(x1i)+"x2i"+str(x2i)+".png",dpi=600,bbox_inches='tight')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plot positions versus all time
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(times,x_2,color='k',label="Mass Two")
    ax1.plot(times,x_1,color='r',label="Mass One")
    ax1.set_title("Positions: "+r'$\delta t = $'+str(round(times[1]-times[0],4))+" "+r'$m_1 = $'+str(m1)+" "+r'$v_1 = $'+str(v1i)+" "+r'$m_2 = $'+str(m2)+" "+r'$v_2 = $'+str(v2i),style='italic')
    ax1.set_ylabel("Position of Masses "+r'$\left(\frac{xmg}{E}\right)$',style='italic')
    ax1.set_xlabel("Time "+r'$\left(\frac{tgm^{1/2}}{E^{1/2}}\right)$',style='italic')
    ax1.legend(loc='upper right')
    f1.show()
    f1.savefig("Positions-MassOne"+str(m1)+"MassTwo"+str(m2)+"x1i"+str(x1i)+"x2i"+str(x2i)+".png",dpi=600,bbox_inches='tight')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plot positions versus small window
    f2,ax2 = plt.subplots()
    f2.tight_layout()
    ax2.plot(times[0:20000],x_2[0:20000],color='k',label="Mass Two")
    ax2.plot(times[0:20000],x_1[0:20000],color='r',label="Mass One")
    ax2.set_title("Positions Zoomed: "+r'$\delta t = $'+str(round(times[1]-times[0],4))+" "+r'$m_1 = $'+str(m1)+" "+r'$v_1 = $'+str(v1i)+" "+r'$m_2 = $'+str(m2)+" "+r'$v_2 = $'+str(v2i),style='italic')
    ax2.set_ylabel("Position of Masses "+r'$\left(\frac{xmg}{E}\right)$',style='italic')
    ax2.set_xlabel("Time "+r'$\left(\frac{tgm^{1/2}}{E^{1/2}}\right)$',style='italic')
    ax2.legend(loc='upper right')
    f2.show()
    f2.savefig("PositionsZoomed-MassOne"+str(m1)+"MassTwo"+str(m2)+"x1i"+str(x1i)+"x2i"+str(x2i)+".png",dpi=600,bbox_inches='tight')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plot Correlation Functions
    f3,ax3 = plt.subplots()
    f3.tight_layout()
    ax3.plot(lag,autocorrelation2,color='k',label="Mass Two")
    ax3.plot(lag,autocorrelation1,color='r',label="Mass One")
    ax3.plot(lag,np.zeros(len(lag)),color='k')
    ax3.set_title("AutoCorrelation: "+r'$\delta t = $'+str(round(times[1]-times[0],4))+" "+r'$m_1 = $'+str(m1)+" "+r'$v_1 = $'+str(v1i)+" "+r'$m_2 = $'+str(m2)+" "+r'$v_2 = $'+str(v2i),style='italic')
    ax3.set_ylabel("AutoCorrelation",style='italic')
    ax3.set_xlabel("Time "+r'$\left(\frac{tgm^{1/2}}{E^{1/2}}\right)$',style='italic')
    ax3.legend(loc='upper right')
    f3.show()
    f3.savefig("Correlation-MassOne"+str(m1)+"MassTwo"+str(m2)+"x1i"+str(x1i)+"x2i"+str(x2i)+".png",dpi=600,bbox_inches='tight')
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates Lyapunov exponents
def question7():
    dif = 10**(-6)
    #first two are chaotic comparisons, second two are normal comparisons
    conditions = [[1,9,1,0,3.0,0],
                  [1,9,1,0,3+dif,0],
                  [1,9,.01,0,.75,0],
                  [1,9,.01,0,.75+dif,0]]
    x21,time1 = [],[]
    x22,time2 = [],[]
    x23,time3 = [],[]
    x24,time4 = [],[]
    for c in conditions:
        m1,m2,x1i,v1i,x2i,v2i = c[0],c[1],c[2],c[3],c[4],c[5]
        #Calculate Initial Energy
        m = m1 + m2
        E = (1/2)*(m1*v1i**2 + m2*v2i**2) + m1*9.807*x1i + m2*9.807*x2i
        print(E)
        g = 9.807
        x_scale = E/(m*g)
        v_scale = (E/m)**(1/2)
        t_scale = (E/(m*g**2))**(1/2)
        Eini = 0.5*(m1*(v1i/v_scale)**2 + m2*(v2i/v_scale)**2)/m + (m1*(x1i/x_scale) + m2*(x2i/x_scale))/m
        print ('Initial Energy = ',Eini)
        if abs(Eini - 1.0) > 1.e-6:
            print ('\nInitial Energy not equal to 1. Stopping execution.\n')
            return
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Find data for Poincar'e Sections
        x2ps = []
        v2ps = []
        x1,v1,x2,v2 = x1i/x_scale,v1i/v_scale,x2i/x_scale,v2i/v_scale
        i = 50 #number of collisions
        ttofall = timetofall(x1,v1)
        ttocollide  = float
        time = 0
        while i > 0:
            isbouncing=True
            while isbouncing:
                #Balls fall, update points accordingly
                ttofall = timetofall(x1,v1)
                time = time + ttofall
                x1,v1,x2,v2 = update(x1,v1,x2,v2,ttofall)
                x1, v1 = 0, -1*v1
                #Where would the balls be if they collide. Is it below x=0? If so bounce again
                ttocollide = timetocollide(x1,v1,x2,v2)
                x1t,v1t,x2t,v2t = update(x1,v1,x2,v2,ttocollide)
                #Does the ball bounce again?
                if x1t<0:
                    ttofall = timetofall(x1,v1)
                else:
                    isbouncing = False
            #Balls collide, update points accordingly        
            time = time + ttocollide
            x1,v1,x2,v2 = update(x1,v1,x2,v2,ttocollide)
            v1,v2 = collision(m1,v1,m2,v2)
            x2ps.append(x2)
            v2ps.append(v2)
            i-=1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Find data for x1,v1,x2,v2 positions at equal time intervals across time it took to find 1000 collisions
        steps = 25000
        x_1 = np.zeros(steps)
        x_2 = np.zeros(steps)
        v_1 = np.zeros(steps)
        v_2 = np.zeros(steps)
        times = np.linspace(0,time,steps,endpoint=True)
        x_1[0],v_1[0],x_2[0],v_2[0] = x1i/x_scale,v1i/v_scale,x2i/x_scale,v2i/v_scale
        count = 0
        while count<len(times)-1:
            count+=1
            #Adjust current values for collisions and bounces if they occured between time steps
            event=True
            x1,v1,x2,v2 = x_1[count-1],v_1[count-1],x_2[count-1],v_2[count-1]
            while event:
                #print(count,x1,v1,x2,v2)
                timepassed=0
                #Previous timesteps values, used to accurately find current values
                #Does ball one hit the floor
                if x1<=0:
                    #When did ball one hit the floor between time[count] and time[count-1]
                    ttofall = timetofall(x1,v1)
                    #Update values across delta-time = ttofall
                    x1,v1,x2,v2 = update(x1,v1,x2,v2,ttofall)
                    x1,v1 = 0, -1*v1
                    #Increased timepassed
                    timepassed+=ttofall
                    #Update values at 
                    x1,v1,x2,v2 = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
                #Have the balls collided    
                elif x1>x2:
                    #When did the balls collide?
                    ttocollide = timetocollide(x1,v1,x2,v2)
                    #update values at across delta-time = ttocollide
                    x1,v1,x2,v2 = update(x1,v1,x2,v2,ttocollide)
                    v1,v2 = collision(m1,v1,m2,v2)
                    #Increase timepassed
                    timepassed+=ttocollide
                    #Update values at 
                    x1,v1,x2,v2 = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
                #If the balls are finished colliding/bouncing then update values correctly and leave loop
                else:
                    #Update accurate values
                    x_1[count],v_1[count],x_2[count],v_2[count] = update(x1,v1,x2,v2,times[count]-times[count-1]-timepassed)
                    event=False
        if(conditions.index(c)==0):
            x21,time1 = x_2,times
        elif(conditions.index(c)==1):
            x22,time2 = x_2,times
        elif(conditions.index(c)==2):
            x23,time3 = x_2,times
        else:
            x24,time4 = x_2,times
    #Create difference arrays between the two chaotic and two "normal" systems
    x2_c_dif,x2_n_dif = [],[]
    for i in list(range(0,len(x21))):
        x2_c_dif.append(np.abs(x22[i] - x21[i])/dif)
        x2_n_dif.append(np.abs(x24[i] - x23[i])/dif)
        
    popt2, pcov2 = curve_fit(func, time2, x2_c_dif, maxfev=100000)
    popt4, pcov4 = curve_fit(func, time4, x2_n_dif, maxfev=100000)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Regular Line Plot
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(time2,x2_c_dif,color='r',label="Actual"+r'$m_2$')
    ax1.plot(time2,func(time2,*popt2),color='r',linestyle='dotted',label='fit: b='+str(round(popt2[0],4)))
    ax1.set_title("Chaotic Lyapunov ",style='italic')
    ax1.set_ylabel("Difference "+r'$\frac{\delta Z(t)}{\delta Z(0)}$',style='italic')
    ax1.set_xlabel("Time "+r'$\left(\frac{tgm^{1/2}}{E^{1/2}}\right)$',style='italic')
    ax1.legend(loc='upper right')
    f1.show()
    f1.savefig("ChaoticLyapunov.png",dpi=600,bbox_inches='tight')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Scatter Plot
    f2,ax2 = plt.subplots()
    f2.tight_layout()
    ax2.plot(time4,x2_n_dif,color='r',label="Actual "+r'$m_2$')
    ax2.plot(time4,func(time4,*popt4),color='r',linestyle='dotted',label='fit: b='+str(round(popt4[0],4)))
    ax2.set_title("Normal Lyapunov ",style='italic')
    ax2.set_ylabel("Difference "+r'$\frac{\delta Z(t)}{\delta Z(0)}$',style='italic')
    ax2.set_xlabel("Time "+r'$\left(\frac{tgm^{1/2}}{E^{1/2}}\right)$',style='italic')
    ax2.legend(loc='upper right')
    f2.show()
    f2.savefig("NormalLyapunov.png",dpi=600,bbox_inches='tight')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creates overlaying Poincar'e sections
def multipoincare():
    conditions = [[1,1,1,0,3,0],
                  [1,5,1,0,3,0],
                  [1,1,.9,0,3.1,.1],
                  [1,1,.8,0,3.2,.2],
                  [1,1,.7,-.1,3.2,.5],
                  [1,1,.5,-.5,3.5,.8],
                  [1,1,.3,1,3,-.2],
                  [1.01,.99,1,0,3,.2],
                  [1,4,1,0,3.1,-.1],
                  [1,3,1.5,0,3.5,.1],
                  [1,1,.9,.9,3,.1],
                  [1,3,1,0,3,0]]
    f,ax = plt.subplots()
    f.tight_layout()
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    for c in conditions:
        m1,m2,x1i,v1i,x2i,v2i = c[0], c[1], c[2], c[3], c[4], c[5]
        #Calculate Initial Energy
        m = m1 + m2
        E = (1/2)*(m1*v1i**2 + m2*v2i**2) + m1*9.807*x1i + m2*9.807*x2i
        print(E)
        g = 9.807
        x_scale = E/(m*g)
        v_scale = (E/m)**(1/2)
        t_scale = (E/(m*g**2))**(1/2)
        Eini = 0.5*(m1*(v1i/v_scale)**2 + m2*(v2i/v_scale)**2)/m + (m1*(x1i/x_scale) + m2*(x2i/x_scale))/m
        print ('Initial Energy = ',Eini)
        if abs(Eini - 1.0) > 1.e-6:
            print ('\nInitial Energy not equal to 1. Stopping execution.\n')
            return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Find data for Poincar'e Sections
        x2ps = []
        v2ps = []
        x1,v1,x2,v2 = x1i/x_scale,v1i/v_scale,x2i/x_scale,v2i/v_scale
        i = 1000 #number of collisions
        ttofall = timetofall(x1,v1)
        ttocollide  = float
        time = 0
        while i > 0:
            isbouncing=True
            while isbouncing:
                #Balls fall, update points accordingly
                ttofall = timetofall(x1,v1)
                time = time + ttofall
                x1,v1,x2,v2 = update(x1,v1,x2,v2,ttofall)
                x1, v1 = 0, -1*v1
                #Where would the balls be if they collide. Is it below x=0? If so bounce again
                ttocollide = timetocollide(x1,v1,x2,v2)
                x1t,v1t,x2t,v2t = update(x1,v1,x2,v2,ttocollide)
                #Does the ball bounce again?
                if x1t<0:
                    ttofall = timetofall(x1,v1)
                else:
                    isbouncing = False
            #Balls collide, update points accordingly        
            time = time + ttocollide
            x1,v1,x2,v2 = update(x1,v1,x2,v2,ttocollide)
            v1,v2 = collision(m1,v1,m2,v2)
            x2ps.append(x2)
            v2ps.append(v2)
            i-=1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Create all Graphs
        #Plot Poincar'e Sections    
        ax.plot(x2ps,v2ps,alpha=.4,label=conditions.index(c)+1)
        ax1.scatter(x2ps,v2ps,alpha=.4,label=conditions.index(c)+1)
    ax.set_title("Poincar´e Sections: Multiple Overlayed",style='italic')
    ax.set_ylabel("Velocity of "+r'$Mass_2 $'+r'$\left(\frac{v_2E^{1/2}}{m^{1/2}}\right)$',style='italic')
    ax.set_xlabel("Position of "+r'$Mass_2 $'+r'$\left(\frac{x_2E}{mg}\right)$',style='italic')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.legend(loc='upper right')
    f.show()
    f.savefig("Section-MultiConditions"+".png",dpi=600,bbox_inches='tight')
    ax1.set_title("Poincar´e Sections: Multiple Overlayed",style='italic')
    ax1.set_ylabel("Velocity of "+r'$Mass_2 $'+r'$\left(\frac{v_2E^{1/2}}{m^{1/2}}\right)$',style='italic')
    ax1.set_xlabel("Position of "+r'$Mass_2 $'+r'$\left(\frac{x_2E}{mg}\right)$',style='italic')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right')
    f1.show()
    f1.savefig("Section-MultiConditions1"+".png",dpi=600,bbox_inches='tight')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code. 
#m1,m2,x1,v1,x2,v2
#Questions 2-3 ##########################
#chaos_balls(1,.5,1,0,3,0)  
#chaos_balls(1,1,1,0,3,0)   
chaos_balls(1,9,1,0,3,0)
#Questions 5 ###########################
#chaos_balls(1,9,.01,0,.75,0)   
#chaos_balls(1,9,1.3,0,2.6,0) #NON-Chaotic
#Question 6 ############################
#multipoincare()
#Question 7 ##########################
#question7()