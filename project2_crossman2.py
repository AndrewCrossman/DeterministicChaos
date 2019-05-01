# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:32:57 2019

@author: Andrew Crossman
"""
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

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
def chaos_balls():
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
    ax.set_title("Poincar´e Sections: Multiple Overlayed",style='italic')
    ax.set_ylabel("Velocity of "+r'$Mass_2 $'+r'$\left(\frac{v_2E^{1/2}}{m^{1/2}}\right)$',style='italic')
    ax.set_xlabel("Position of "+r'$Mass_2 $'+r'$\left(\frac{x_2E}{mg}\right)$',style='italic')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.legend(loc='upper right')
    f.show()
    f.savefig("Section-MultiCOnditions"+".png",dpi=600,bbox_inches='tight')
chaos_balls()