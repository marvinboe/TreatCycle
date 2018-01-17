#!/usr/bin/env python3
#######################################################################
# overall_growth.py
#Calculate and plot overall average growth under cycling treatment 
#conditions for different treatment durations or strengths.
#
#Copyright 2017 Marvin A. Böttcher
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
########################################################################

import numpy as np
import scipy.integrate 
import scipy.optimize as opti
import matplotlib.pyplot as plt
import plothelpers
import itertools

duration_not_strength=True #Plotting either variable duration or variable treatment strength

epsilon=0.1
mig_spont=0.1 #phenotype migration rate
mig_ind=0.0 #phenotype migration rate

startfrac=0.5
base_growth=1
comp_growth=1/2.

sim_time=900
interval_length=20

base_lh_1=0.75
base_lh_2=1.
base_eps_1=0.
base_eps_2=0.1
base_lw=0.25

test_cycles=50 #number of cycles to calculate average growth from

if duration_not_strength:
    dT=0.3#0.15
    Tmax=2.5
    Tmin=0.1
    Ts_high=np.arange(Tmin,Tmax,dT)
    Ts_low=np.arange(Tmin,Tmax,dT)
else:
    dl=0.1
    lmax=1.21
    lmin=0.6
    lweak=0.25
    Ls_high=np.arange(lmin,lmax,dl)
    Ls_low=np.array([lweak for i in Ls_high])
    Th=1. #treatment interval duration high dosage
    Tl=3. #treatment interval duration low dosage


def f(x,t,l,epsilon,mig_spont=mig_spont,mig_ind=mig_ind):
    """ function defining the model dx/dt=f(x)"""

    x0=x[0]# if x[0] < 1 else 1.-1e-10
    x1=x[1]# if x[1] < 1 else 1.-1e-10
    f0=epsilon
    f1=((base_growth-l-epsilon)*x0+(comp_growth-l)*x1)#*x0+(1/2.-l)*x1
    fbar=f0*x0+f1*x1

    if fbar > 0:
        x0_dot=(f0-fbar)*x0 -mig_ind*fbar*x0 +mig_spont*(x1-x0)
        x1_dot=(f1-fbar)*x1 +mig_ind*fbar*x0 -mig_spont*(x1-x0)
    else:
        x0_dot=(f0-fbar)*x0 -mig_ind*fbar*x1 +mig_spont*(x1-x0)
        x1_dot=(f1-fbar)*x1 +mig_ind*fbar*x1 -mig_spont*(x1-x0)
    return np.array([x0_dot,x1_dot])

def calc_av_fitness(x,l,epsilon,mig_spont=mig_spont,mig_ind=mig_ind):
    """ function defining the model dx/dt=f(x)"""
    x0=x[0]# if x[0] < 1 else 1.-1e-10
    x1=x[1]# if x[1] < 1 else 1.-1e-10
    f0=epsilon
    f1=(base_growth-l-epsilon)*x0+(comp_growth-l)*x1
    fbar=f0*x0+f1*x1
    return fbar


def fitfunc(t,a,n0):
    return n0*np.exp(a*t)

def fitfunc_lin(t,a,n0):
    return a*t+n0


def no_cells(av_fitness,dt=0.1, N0=1000):
    N=N0
    returnlist=[]
    for l in av_fitness:
        returnlist.append(N)
        dN=N * (np.exp(l*dt)-1)
        N=N+dN
    return np.array(returnlist)



def integrate(f,t_start=0,t_max=100,t_step=0.01,x0=np.array([1/2.,1/2.]),**kwargs):
    t=np.arange(t_start,t_max,t_step)
    sol=scipy.integrate.odeint(f,x0,t,rtol=1e-11,atol=1e-11,**kwargs) #1.49012e-8
    if (np.sum(sol[-1])>1) or np.any(sol[-1] < 0.):
        error=np.sum(sol[-1])
        # print("error in integration" ,sol[-1])
    return t,sol



def find_increase(ts=1,tw=1,ls=1.,lw=0.25,epsilon=epsilon,mig_spont=mig_spont,mig_ind=mig_ind,startfrac=startfrac):
    """calculates overall growth by fitting linear curve to logplot."""

    Ts=ts*interval_length
    Tw=tw*interval_length
    Tmax=sim_time #(Ts+Tw)*
    
    ls=itertools.cycle([ls,lw])
    durations_cycle=itertools.cycle([Ts,Tw])
    
    sol=[]
    av_fitness=[]
    t=[]
    
    l=next(ls)
    
    def fun(x,t):
        return f(x,t,l,epsilon,mig_spont,mig_ind)
    
    dur=next(durations_cycle)
    
    time,solution=integrate(fun,x0=[startfrac,1-startfrac],t_max=dur)
    # print(solution)
    av_f=np.array([calc_av_fitness(x,l,epsilon) for x in solution])
    av_fitness.append(av_f)
    sol.append(solution)
    t.append(time)
    
    for i in range(test_cycles):
        dur=next(durations_cycle)
        l=next(ls)
    
        def fun(x,t):
            return f(x,t,l,epsilon,mig_spont,mig_ind)
    
        if (abs(solution[-1].any()) > 1.):
            print ("error")
            break
        else:
            time,solution=integrate(fun,x0=solution[-1],t_max=time[-1]+dur,t_start=time[-1],mxstep=5000000)
        av_f=np.array([calc_av_fitness(x,l,epsilon) for x in solution])
        av_fitness.append(av_f)
        sol.append(solution)
        t.append(time)
    
    sol=np.concatenate(sol)
    frac=np.average(sol[:,0])
    av_fitness=np.concatenate(av_fitness)
    tc=np.concatenate(t)
    
    timestep=(tc[-1]-tc[-2])
    logcells=np.log(no_cells(av_fitness,dt=timestep))
    
    print(frac)
    try:
        popt,pcov=opti.curve_fit(fitfunc_lin,tc,logcells,p0=(0.01,1000))
    except:
        return np.NaN
    return popt[0],frac
find_increase=np.vectorize(find_increase)

plothelpers.latexify(columns=2,fig_height=2)
colors=itertools.cycle(["blue","orange"])
fig,axes=plt.subplots(1,2)
ax=axes[0]
ax2=axes[1]
ax.margins(0.05)
ms=6.

if duration_not_strength:

    color=next(colors)
    data,frac=find_increase(ts=Ts_high,tw=Ts_low,ls=base_lh_2,lw=base_lw,epsilon=base_eps_2)
    plot=ax.plot(Ts_high,data,linestyle='None',marker='o',
            ms=ms,label="$\epsilon="+str(base_eps_2)+"$",color=color)#"$\lambda_H="+str(base_lh_2)+
    ax2.plot(Ts_high,frac,linestyle='None',marker='o',
            ms=ms,label="$\epsilon="+str(base_eps_2)+"$",color=color)#"$\lambda_H="+str(base_lh_2)+

    color=next(colors)
    data,frac=find_increase(ts=Ts_high,tw=Ts_low,ls=base_lh_2,lw=base_lw,epsilon=base_eps_1)
    plot=ax.plot(Ts_high,data,linestyle='None',marker='v',
            ms=ms,label="$\epsilon="+str(base_eps_1)+"$",color=color) #"$\lambda_H="+str(base_lh_2)+
    ax2.plot(Ts_high,frac,linestyle='None',marker='v',
            ms=ms,label="$\epsilon="+str(base_eps_1)+"$",color=color) #"$\lambda_H="+str(base_lh_2)+


else:
    color=next(colors)
    data,frac=find_increase(ts=Th,tw=Tl,ls=Ls_high,lw=Ls_low,epsilon=base_eps_1)
    label="$\epsilon="+str(base_eps_1)+"$"
    plot=ax.plot(Ls_high,data,linestyle='None',marker='o',
            ms=ms,label=label,color=color)
    ax2.plot(Ls_high,frac,linestyle='None',marker='v',
            ms=ms,label="$\epsilon="+str(base_eps_1)+"$",color=color) #"$\lambda_H="+str(base_lh_2)+


    color=next(colors)
    data,frac=find_increase(ts=Th,tw=Tl,ls=Ls_high,lw=Ls_low,epsilon=base_eps_2)
    label="$\epsilon="+str(base_eps_2)+"$"
    plot=ax.plot(Ls_high,data,linestyle='None',marker='o',
            ms=ms,label=label,color=color)
    ax2.plot(Ls_high,frac,linestyle='None',marker='v',
            ms=ms,label="$\epsilon="+str(base_eps_2)+"$",color=color) #"$\lambda_H="+str(base_lh_2)+

ax.axis('tight')
ax.autoscale(enable=True, axis='x', tight=True)

if duration_not_strength:
    ax.set_xlim(xmin=-0.001)

    ax.set_xlabel("Treatment cycle length")
    ax.set_ylabel("Overall growth")
    ax2.set_xlabel("Treatment cycle length")
    ax2.set_ylabel("av. D cell fraction")
else :
    ax.set_xlabel("Strong treatment strength")
    ax.set_ylabel("Overall growth")

ax2.legend(loc=0,numpoints=1,borderpad=0.3,borderaxespad=0.2, fontsize = 'small',labelspacing=0.3)
fig.tight_layout(pad=0.6)
# fig.subplots_adjust(top=0.98,bottom=0.2,left=0.185,right=0.98)
plt.show()





