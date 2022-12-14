# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 00:29:04 2022

@author: 85176
"""

import jdata as jd
# from scipy import stats
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sympy import *
from IPython.core.display import Latex
import sys      
from iminuit import Minuit        
import statistics                                 # Module to see files and folders in directories
from IPython.core.display import Latex




sys.path.append('C:/Users/85176/AppStat2022/External_Functions')
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure
sys.path.append('C:/Users/85176/AppStat2022/Week0')
from WeightedMeanSigmaChi2_original import mean_weighted
from WeightedMeanSigmaChi2_original import err_weighted
from WeightedMeanSigmaChi2_original import chi2_owncalc

def lprint(*args,**kwargs):
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)


ball_on_incline_data = jd.load('ball_on_incline_data.json')


d_ball = ball_on_incline_data.get('ball diameters')
d_ball_err = ball_on_incline_data.get('ball diameters errors')

angle = ball_on_incline_data.get('theta')
exp_theta = np.array(angle.get('phone measuremnet'))
theta_error = 0.1;theta_error = np.std(exp_theta)
# theta = np.mean(theta)
t_peak = ball_on_incline_data.get('gate peak times')
d_rail=np.array([5.9,5.8,5.9,6.0,5.8,5.6])/1000
d_rail_error = np.std(d_rail); d_rail = np.mean(d_rail)

d_hole = ball_on_incline_data.get('hole diameters')


gate_positions = np.array([21.51,38.15,55.78,72.21,89.83])/100



A = np.array([21.85,22.05,22.06]);B = np.array([89.6,89.1,89.15])
theta_refer = np.arctan(A/B)*180/np.pi


d_big = d_ball.get('big')
d_big = np.array(d_big)/1000
d_big_error = 0.1/1000;d_big_error = np.std(d_big)
big_exp = t_peak.get('big ball').get('peak times')



d_small = d_ball.get('small')
d_small = np.array(d_small)/1000
d_small_error = 0.1/1000;d_small_error = np.std(d_small)
small_exp = t_peak.get('small ball').get('peak times')



peak_centers = np.zeros([9,5])


# Define variables:
g,a,theta,delta_theta,D,d = symbols("g,a,theta, delta_theta, D, d")
dg,da,dtheta,ddelta_theta,dD,dd = symbols("sigma_g,sigma_a, sigma_theta, sigma_delta_theta, sigma_D, sigma_d")
g = a/sin((theta+delta_theta)/180*pi)*(1+2/5*D**2/(D**2-d**2))
dg = sqrt((g.diff(a) * da)**2 + (g.diff(theta) * dtheta)**2+(g.diff(delta_theta) * ddelta_theta)**2+(g.diff(D) * dD)**2+(g.diff(d) * dd)**2)
fg = lambdify((a,theta,delta_theta,D,d),g)
fdg = lambdify((a,da,theta,dtheta,delta_theta,ddelta_theta,D,dD,d,dd),dg)
aaaa = latex(Eq(symbols('sigma_g'), dg))





vdelta_theta, vddelta_theta = -0.1,0.1

vd, vdd = d_rail,d_rail_error

j = 0
colors = ['red','blue','green']

for per_exp in range(0,9):
    peaks = small_exp[per_exp]
    for peak in range(0,5):
        peak_range = peaks[peak]
        peak_center = statistics.median(peak_range)
        peak_centers[per_exp,peak]= peak_center
    peak_centers[per_exp,:] = peak_centers[per_exp,:]-peak_centers[per_exp,0]
    

Exp1 = peak_centers[0,:];
Exp2 = peak_centers[3,:];
Exp3 = peak_centers[8,:];


Peak_centers = np.zeros([3,5])
Peak_centers[0,:] = Exp1
Peak_centers[1,:] = Exp2
Peak_centers[2,:] = Exp3

def time_distance(t,u0,a):
    t = np.array(t)
    y = u0*t+0.5*a*t**2
    return y


y = gate_positions-gate_positions[0];
sy = 2.5/1000
a_value = np.zeros(3);a_error = np.zeros(3);
gg = np.zeros(3);vg = np.zeros(3);vdg = np.zeros(3);
a_chi2 = np.zeros(3);a_probs = np.zeros(3)
u0_value = np.zeros(3)
fig, ax = plt.subplots(figsize=(10, 6))  # figsize is in inches



for i in range(0,3):
    t = Peak_centers[i,:]
    st = 0.0002
    
    ax.errorbar(t, y, xerr=st, yerr=sy, label = f'Experimenter {i+1}',fmt='.', mec=colors[i], mfc = colors[i], ecolor='k', elinewidth=1, capsize=1, capthick=1)
    ax.legend(loc='upper left', numpoints=1)
    
    
    chi2fit = Chi2Regression(time_distance, t, y, sy)
    minuit_chi2 = Minuit(chi2fit, u0 = 0, a = 1)
    minuit_chi2.errordef = 1.0     # This is the definition for ChiSqaure fits
    minuit_chi2.migrad()           # This is where the minimisation is carried out! Put ";" at the end to void output

    if (not minuit_chi2.fmin.is_valid) :
        print("  WARNING: The ChiSquare fit DID NOT converge!!! ")    
    
    a_value[i] = minuit_chi2.values['a']
    u0_value[i] = minuit_chi2.values['u0']
    a_error[i] = minuit_chi2.errors['a']

    print(f"Fit value: a = {a_value[i]:.5f} +/- {a_error[i]:.5f}")


    chi2_value = minuit_chi2.fval
    Ndof_value = len(Exp1) - minuit_chi2.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) # The chi2 probability given N_DOF degrees of freedom
    print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")

    d = { 'Experimenter': i+1,
          'a':     [minuit_chi2.values['a'], minuit_chi2.errors['a']],
          'u0':     [minuit_chi2.values['u0'], minuit_chi2.errors['u0']],
          'Chi2':     chi2_value,
          'Prob':     Prob_value,
        }
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.23, 0.98-0.2*i, text, ax, fontsize=9)
    
    if Prob_value > 0.005:
        
        a_value[j] = minuit_chi2.values['a']
        u0_value[j] = minuit_chi2.values['u0']
        a_error[j] = minuit_chi2.errors['a']
    
        a_chi2[j] = minuit_chi2.fval
        a_probs[j] = stats.chi2.sf(chi2_value, Ndof_value)
        aa = minuit_chi2.values['a']
        u00 = minuit_chi2.values['u0']
    
        tt = 0.01*np.array(range(0,80,1))
        yy = time_distance(tt,u00,aa)
        ax.plot(tt,yy,'-',color = colors[i])
        ax.plot(t,y,'.',color = colors[i])
        ax.set(xlabel="Time t (s)", # the label of the y axis
           ylabel="Travel distance y (m)",  # the label of the y axis
           title='Small Ball   '+r'$y = u_0t+\frac{1}{2}at^2, \sigma_y = 2.5x10^{-3}m, \sigma_t = 2x10^{-4}s$', 
           ylim=[0.0,None]) # Setting the miminum to zero
        
        
      

    
        va, vda = a_value[j],a_error[j]
        vD, vdD = d_small[j],d_small_error
        vtheta, vdtheta = exp_theta[j],theta_error


        vg[j] = fg(va,vtheta,vdelta_theta,vD,vd)
        vdg[j]= fdg(va,vda,vtheta,vdtheta,vdelta_theta,vddelta_theta,vD,vdD,vd,vdd)
    
        j = j+1
        



g123 = mean_weighted(vg,vdg)
g456 = err_weighted(vg,vdg)
print(f"Value: g = {g123:.5f} +/- {g456:.5f}")