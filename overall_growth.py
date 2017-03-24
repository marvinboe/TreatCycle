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

duration_not_strength=True

epsilon=0.04
mig=0.01

sim_time=900
interval_length=20


if duration_not_strength:
    dT=0.2
    Tmax=3.0
    X=np.arange(dT,Tmax,dT)
    Y=np.arange(dT,Tmax,dT)
else:
    dl=0.1
    lmax=1.2+0.01
    lmin=0.6
    lweak=0.25
    X=np.arange(lmin,lmax,dl)
    Y=np.array([lweak for i in X])
    interval_1=1. #treatment interval duration 
    interval_2=3. #treatment interval duration 

cmap = plt.get_cmap('viridis')
indices = np.linspace(0, cmap.N, len(X)+1)
my_colors = itertools.cycle([cmap(int(i)) for i in indices])

def set_payoff_array(e,l):
    a11=a12=e
    # a12=a13=0
    a21=1.-l -e
    a22=1./2.-l
    A=np.array([[a11,a12],
            [a21,a22]])
    return A

def set_mutation_matrix(mutation_to_dorm,mutation_from_dorm=-1.):
    if mutation_from_dorm<0.:
        mutation_from_dorm=mutation_to_dorm
    a12=mutation_to_dorm
    a21=mutation_from_dorm
    a11=1-a21
    a22=1-a12
    A=np.array([[a11,a12],
            [a21,a22]])
    return A


def f(x,t,A,M):
    x0=x[0]
    x1=x[1]
    phi=(-x.dot(A.dot(x)))
    x0_dot=M.dot(x*A.dot(x))[0]+x0*phi
    x1_dot=M.dot(x*A.dot(x))[1]+x1*phi
    return np.array([x0_dot,x1_dot])


def fitfunc(t,a,n0):
    return n0*np.exp(a*t)

def fitfunc_lin(t,a,n0):
    return a*t+n0


def calc_av_fitness(x,A):
    return (x.dot(A.dot(x)))

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



def find_increase(ts=1,tw=1,ls=1.,lw=0.25,epsilon=epsilon,mig=0.0001):
    """calculates overall growth by fitting linear curve to logplot."""
    e=epsilon

    Ts=ts*interval_length
    Tw=tw*interval_length
    Tmax=sim_time #(Ts+Tw)*
    
    M=set_mutation_matrix(mig)
    
    payoff_matrices=itertools.cycle([set_payoff_array(e,ls),set_payoff_array(e,lw)])
    durations_cycle=itertools.cycle([Ts,Tw])
    
    sol=[]
    av_fitness=[]
    t=[]
    
    payoff_matrix=next(payoff_matrices)
    
    def fun(x,t):
        return f(x,t,payoff_matrix,M)
    
    dur=next(durations_cycle)
    
    time,solution=integrate(fun,x0=[0.1,0.9],t_max=dur)
    # print(solution)
    av_f=np.array([calc_av_fitness(x,payoff_matrix) for x in solution])
    av_fitness.append(av_f)
    sol.append(solution)
    t.append(time)
    
    # av_fitness=[]
    while (time[-1]<Tmax):
        dur=next(durations_cycle)
        payoff_matrix=next(payoff_matrices)
    
        def fun(x,t):
            return f(x,t,payoff_matrix,M)
    
        if (abs(solution[-1].any()) > 1.):
            print ("error")
            break
        else:
            time,solution=integrate(fun,x0=solution[-1],t_max=time[-1]+dur,t_start=time[-1],mxstep=5000000)
        av_f=np.array([calc_av_fitness(x,payoff_matrix) for x in solution])
        av_fitness.append(av_f)
        sol.append(solution)
        t.append(time)
    
    sol=np.concatenate(sol)
    av_fitness=np.concatenate(av_fitness)
    tc=np.concatenate(t)
    
    timestep=(tc[-1]-tc[-2])
    logcells=np.log(no_cells(av_fitness,dt=timestep))
    
    # print(logcells)
    try:
        popt,pcov=opti.curve_fit(fitfunc_lin,tc,logcells,p0=(0.01,1000))
    except:
        return np.NaN
        
    return popt[0]
find_increase=np.vectorize(find_increase)

plothelpers.latexify(columns=1)
fig,ax=plt.subplots(1,1)
ax.margins(0.05)
fig.subplots_adjust(top=0.98,bottom=0.2,left=0.185,right=0.98)
ms=6.

if duration_not_strength:
    data=find_increase(ts=X,tw=Y,ls=0.75,lw=0.25,epsilon=epsilon,mig=0.01)
    data2=find_increase(ts=X,tw=Y,ls=1.,lw=0.25,epsilon=epsilon,mig=0.01)
    plot=ax.plot(X,data,linestyle='None',marker='o',ms=ms,label="$\lambda_H=0.75$")
    plot=ax.plot(X,data2,linestyle='None',marker='v',ms=ms,label="$\lambda_H=1$")
else:
    data=find_increase(ts=interval_1,tw=interval_1,ls=X,lw=Y,epsilon=epsilon,mig=mig)
    data2=find_increase(ts=interval_2,tw=interval_2,ls=X,lw=Y,epsilon=epsilon,mig=mig)
    plot=ax.plot(X,data,linestyle='None',marker='o',ms=ms,label="$T_H="+str(interval_1)+"$")
    plot=ax.plot(X,data2,linestyle='None',marker='v',ms=ms,label="$T_H="+str(interval_2)+"$")
    # ax.set_yscale('log')
    # ax.set_ylim(ymin=0.0,ymax=0.1)

ax.axis('tight')

if duration_not_strength:
    ax.set_xlabel("Treatment cycle length")
    ax.set_ylabel("Overall growth")
else :
    ax.set_xlabel("Strong treatment strength")
    ax.set_ylabel("Overall growth")

ax.legend(loc=0,numpoints=1,borderpad=1,borderaxespad=1)
plt.show()





