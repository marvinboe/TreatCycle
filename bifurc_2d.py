#!/usr/bin/env python3
#######################################################################
# bifurc_2d.py
#Plot bifurcation diagram for treatment strehgtn in EGT model 
#of Glioblastoma.
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

import sys
import numpy as np
import matplotlib.pyplot as plt
import fractions
import scipy.optimize as opti
import plothelpers
import itertools

lmax=1.35
l=np.linspace(0,lmax,1000)
e=np.linspace(0,0.5,3)
switching=0.1
e=[0.01]


def set_payoff_array(e,l):
    a11=a12=e
    # a12=a13=0
    a21=1.-l -e
    a22=1./2.-l
    A=np.array([[a11,a12],
            [a21,a22]])
    return A

def calc_av_fitness(x,A):
    return (x.dot(A.dot(x)))

def av_fitness(e,l):
    x=np.array([d(e,l),1.-d(e,l)])
    return calc_av_fitness(x,set_payoff_array(e,l))
av_fitness=np.vectorize(av_fitness)

def ex_av_fitness(x,e,l):
    x=np.array(x)
    return calc_av_fitness(x,set_payoff_array(e,l))


def f(x,e,l,m):
    xd=x[0]
    xag=x[1]
    la=l
    xd_return=(1/2)*(-2*e*(-1 + m)*xd*(xag + xd) + m*xag*(xag - 2*la*xag - 2*(-1 + e + la)*xd) + xd*((-1 + 2*la)*xag**2 + 2*(-1 + la)*xag*xd - 2*e*xd**2))    
    xag_return=(1/2)*((-1 + 2*la)*xag**2*(-1 + m + xag) + 2*xag*(e*(-1 + 2*m) + (-1 + la)*(-1 + m + xag))*xd + 2*e*(m - xag)*xd**2)
    return np.array([xd_return,xag_return])

def eigenvalues(x,e,l,m):
    """EV's of Jacobi matrix, calculated by Mathematica"""
    xd=x[0]
    xag=x[1]
    la=l
    ev1=(1/8)*(4*xag + 4*e*xag - 8*la*xag - 8*e*m*xag + 4*la*m*xag - \
   8*xag**2 + 16*la*xag**2 + 4*xd + 4*e*xd - 4*la*xd - 4*m*xd + \
   4*la*m*xd - 16*xag*xd + \
      16*la*xag*xd - 16*e*xd**2 - \
   np.sqrt((-4*xag - 4*e*xag + 8*la*xag + 8*e*m*xag - 4*la*m*xag + \
        8*xag**2 - 16*la*xag**2 - 4*xd - 4*e*xd + 4*la*xd + \
               4*m*xd - 4*la*m*xd + 16*xag*xd - 16*la*xag*xd + \
        16*e*xd**2)**2 - \
     16*(4*e*xag**2 - 8*e*la*xag**2 - 8*e*m*xag**2 + 16*e*la*m*xag**2 - \
        2*xag**3 - \
               6*e*xag**3 + 8*la*xag**3 + 12*e*la*xag**3 - \
        8*la**2*xag**3 + 12*e*m*xag**3 - 2*la*m*xag**3 - \
        24*e*la*m*xag**3 + 4*la**2*m*xag**3 + 3*xag**4 - \
               12*la*xag**4 + 12*la**2*xag**4 + 8*e*xag*xd - \
        16*e*la*xag*xd - 16*e*m*xag*xd + 32*e*la*m*xag*xd - \
        6*xag**2*xd - 18*e*xag**2*xd + \
               18*la*xag**2*xd + 32*e*la*xag**2*xd - 12*la**2*xag**2*xd + \
        2*m*xag**2*xd + 32*e*m*xag**2*xd - 10*la*m*xag**2*xd - \
        56*e*la*m*xag**2*xd + \
               8*la**2*m*xag**2*xd + 12*xag**3*xd - 36*la*xag**3*xd + \
        24*la**2*xag**3*xd + 8*e*xd**2 - 8*e**2*xd**2 - 8*e*la*xd**2 - \
        16*e*m*xd**2 + \
               16*e**2*m*xd**2 + 16*e*la*m*xd**2 - 4*xag*xd**2 - \
        24*e*xag*xd**2 + 4*e**2*xag*xd**2 + 8*la*xag*xd**2 + \
        36*e*la*xag*xd**2 - 4*la**2*xag*xd**2 + \
               4*m*xag*xd**2 + 32*e*m*xag*xd**2 - 8*e**2*m*xag*xd**2 - \
        8*la*m*xag*xd**2 - 52*e*la*m*xag*xd**2 + 4*la**2*m*xag*xd**2 + \
        12*xag**2*xd**2 + \
               12*e*xag**2*xd**2 - 24*la*xag**2*xd**2 - \
        24*e*la*xag**2*xd**2 + 12*la**2*xag**2*xd**2 - 12*e*xd**3 + \
        4*e**2*xd**3 + 12*e*la*xd**3 + 20*e*m*xd**3 - \
               16*e**2*m*xd**3 - 20*e*la*m*xd**3 + 24*e*xag*xd**3 - \
        24*e*la*xag*xd**3 + 12*e**2*xd**4)))
    ev2=(1/8)*(4*xag + 4*e*xag - 8*la*xag - 8*e*m*xag + 4*la*m*xag - \
   8*xag**2 + 16*la*xag**2 + 4*xd + 4*e*xd - 4*la*xd - 4*m*xd + \
   4*la*m*xd - 16*xag*xd + \
      16*la*xag*xd - 16*e*xd**2 + \
   np.sqrt((-4*xag - 4*e*xag + 8*la*xag + 8*e*m*xag - 4*la*m*xag + \
        8*xag**2 - 16*la*xag**2 - 4*xd - 4*e*xd + 4*la*xd + \
               4*m*xd - 4*la*m*xd + 16*xag*xd - 16*la*xag*xd + \
        16*e*xd**2)**2 - \
     16*(4*e*xag**2 - 8*e*la*xag**2 - 8*e*m*xag**2 + 16*e*la*m*xag**2 - \
        2*xag**3 - \
               6*e*xag**3 + 8*la*xag**3 + 12*e*la*xag**3 - \
        8*la**2*xag**3 + 12*e*m*xag**3 - 2*la*m*xag**3 - \
        24*e*la*m*xag**3 + 4*la**2*m*xag**3 + 3*xag**4 - \
               12*la*xag**4 + 12*la**2*xag**4 + 8*e*xag*xd - \
        16*e*la*xag*xd - 16*e*m*xag*xd + 32*e*la*m*xag*xd - \
        6*xag**2*xd - 18*e*xag**2*xd + \
               18*la*xag**2*xd + 32*e*la*xag**2*xd - 12*la**2*xag**2*xd + \
        2*m*xag**2*xd + 32*e*m*xag**2*xd - 10*la*m*xag**2*xd - \
        56*e*la*m*xag**2*xd + \
               8*la**2*m*xag**2*xd + 12*xag**3*xd - 36*la*xag**3*xd + \
        24*la**2*xag**3*xd + 8*e*xd**2 - 8*e**2*xd**2 - 8*e*la*xd**2 - \
        16*e*m*xd**2 + \
               16*e**2*m*xd**2 + 16*e*la*m*xd**2 - 4*xag*xd**2 - \
        24*e*xag*xd**2 + 4*e**2*xag*xd**2 + 8*la*xag*xd**2 + \
        36*e*la*xag*xd**2 - 4*la**2*xag*xd**2 + \
               4*m*xag*xd**2 + 32*e*m*xag*xd**2 - 8*e**2*m*xag*xd**2 - \
        8*la*m*xag*xd**2 - 52*e*la*m*xag*xd**2 + 4*la**2*m*xag*xd**2 + \
        12*xag**2*xd**2 + \
               12*e*xag**2*xd**2 - 24*la*xag**2*xd**2 - \
        24*e*la*xag**2*xd**2 + 12*la**2*xag**2*xd**2 - 12*e*xd**3 + \
        4*e**2*xd**3 + 12*e*la*xd**3 + 20*e*m*xd**3 - \
               16*e**2*m*xd**3 - 20*e*la*m*xd**3 + 24*e*xag*xd**3 - \
        24*e*la*xag*xd**3 + 12*e**2*xd**4)))
    return np.array([ev1,ev2])


def d(e,l):
    low_fp=0.
    high_fp=1.
    mid_fp=0.
    if e==0.5:
        mid_fp=0.
    else:
        mid_fp=(1. - 2.* e - 2.* l)/( 2.* e - 1.) 

    if stability_mid(e,l):
        return mid_fp
    elif stability_high(e,l):
        return high_fp
    elif stability_low(e,l):
        return low_fp
    else:
        return np.nan
d=np.vectorize(d)

def mid_fp_stable(e,l):
    if e==0.5:
        mid_fp=1.
    else:
        mid_fp=(1. - 2.* e - 2.* l)/( 2.* e - 1.) 
    if mid_fp >=0 and mid_fp <=1. and stability_mid(e,l):
        return mid_fp
    else :
        return np.nan
mid_fp_stable=np.vectorize(mid_fp_stable)

def mid_fp_unstable(e,l):
    if e==0.5:
        mid_fp=np.nan
    else:
        mid_fp=(1. - 2.* e - 2.* l)/( 2.* e - 1.) 


    if mid_fp >= 0 and mid_fp <= 1. and e < 0 and not stability_mid(e,l):
        return mid_fp
    else :
        return np.nan
mid_fp_unstable=np.vectorize(mid_fp_unstable)

def low_fp_stable(e,l):
    if stability_low(e,l):
        return 0.
    else :
        return np.nan
low_fp_stable=np.vectorize(low_fp_stable)

def low_fp_unstable(e,l):
    if not stability_low(e,l):
        return 0.
    else :
        return np.nan
low_fp_unstable=np.vectorize(low_fp_unstable)

def high_fp_stable(e,l):
    if stability_high(e,l):
        return 1.
    else :
        return np.nan
high_fp_stable=np.vectorize(high_fp_stable)

def high_fp_unstable(e,l):
    if not stability_high(e,l):
        return 1.
    else :
        return np.nan
high_fp_unstable=np.vectorize(high_fp_unstable)


def stability_mid(e,l):
    if e==0.5:
        return False
    if e > 0 and (1-4*e+4*e**2 - 3*l+6 *e *l+2*l**2)/(1-2*e) < 0:
        return True
    else : 
        return False
stability_mid=np.vectorize(stability_mid)

def stability_high(e,l):
    if e >= 0 and (1-2*e-l)<=0:
        return True
    else : 
        return False
stability_high=np.vectorize(stability_high)

def stability_low(e,l):
    if l-0.5 <= 0 and e+l-0.5<=0:
        return True
    else : 
        return False
stability_low=np.vectorize(stability_low)


plothelpers.latexify(columns=2,fig_height=2.3)
fig,axes=plt.subplots(1,2,sharex=True)#,figsize=(3.39,2.8)
fig.subplots_adjust(top=0.99, bottom=0.16, right=0.99, left=0.08,wspace=0.22)
ax=axes[0]
ax2=axes[1]
ax.margins(0.06)
ax2.margins(0.06)

linew=1.8
ms=5.5
cmap = plt.get_cmap('viridis_r')
colors=itertools.cycle(["blue","orange"])
for i,epsilon in enumerate(e):
    elabelfrac=fractions.Fraction(epsilon).limit_denominator()
    color=next(colors)

    #plot unstable FP (no phenotype conversion)
    # ax.plot(l,mid_fp_unstable(epsilon,l),"--",lw=linew,color=color)
    # ax.plot(l,low_fp_unstable(epsilon,l),"--",lw=linew,color=color)
    # ax.plot(l,high_fp_unstable(epsilon,l),"--",lw=linew,color=color)

    #plot stable FP (no phenotype conversion)
    ax.plot(l,mid_fp_stable(epsilon,l),color=color,lw=linew,label="$\sigma=0$")#, \epsilon="+str(elabelfrac)+"$")
    ax.plot(l,low_fp_stable(epsilon,l),lw=linew,color=color)
    ax.plot(l,high_fp_stable(epsilon,l),lw=linew,color=color)

    #plot growth rate (no phenotype conversion)
    ax2.plot(l,av_fitness(epsilon,l),lw=linew,color=color,label="$\sigma=0$")#,label="av. fitness")

    ############legend#########
    #change color
    color=next(colors) #cmap(epsilon/np.max(e)/2.)
    ax.plot(np.nan,".",marker='.',color=color,markersize=ms,mew=linew,label="$\sigma="+str(switching)+"$")#, \epsilon="+str(elabelfrac)+"$")
    ax2.plot(np.nan,".",marker='.',color=color,markersize=ms,mew=linew,label="$\sigma="+str(switching)+"$")
    ############end legend#####


    ####calculate and plot FP for model with conversion numerically
    numerical_fps=[]
    for lamb in l[::30]:
        f_collect=[]
        for start_d in np.linspace(0.,1.000000,8):
            sol=opti.root(f,np.array([start_d,1.-start_d]),args=(epsilon,lamb,switching,))
            fp_part=sol.x[0]
            if fp_part < 0. or fp_part > 1.:
                fp_part=np.nan
            #does this FP already exist in list?
            compvalue=(np.sum(np.isclose(f_collect,np.repeat(fp_part,(len(f_collect))),equal_nan=True)))
            #no, not yet:
            if compvalue == 0:
                f_collect.append(fp_part)
                evs=eigenvalues([fp_part,1.-fp_part],epsilon,lamb,switching)
                if np.all(evs<0.):
                    #all EVs<0 -> FP is stable!
                    numerical_fps.append([lamb,fp_part])
                    ### plot numerical FPs
                    lastpointplot=ax.plot(lamb,fp_part,marker='.',color=color,markersize=ms,mew=linew)
                    ax2.plot(lamb,ex_av_fitness([fp_part,1.-fp_part],epsilon,lamb),marker='.',markersize=ms,color=color,mew=linew)
                # else:
                #    #not stable
                #     ax.plot(lamb,fp_part,marker='_',color=color,markersize=ms,mew=linew)
        
 
ax.set_xlabel("Treatment cost $\lambda$")
ax2.set_xlabel("Treatment cost $\lambda$")
ax.set_ylabel("D cell rel. freq.")
ax2.set_ylabel("Growth rate")
leg=ax2.legend(loc=0)


#######draw arrows################
arrow_scale=0.8
l=0.75
ax.arrow(l,0,0,mid_fp_stable(epsilon,l)*arrow_scale)
ax.arrow(l,1,0,-(1-mid_fp_stable(epsilon,l))*arrow_scale)

arrow_scale=0.9
l=1.05
ax.arrow(l,0,0,high_fp_stable(epsilon,l)*arrow_scale)

color=next(colors)
arrow_scale=0.93
l=.25
ax.arrow(l,1,0,-(1-low_fp_stable(epsilon,l))*arrow_scale,color=color)


color=next(colors)
arrow_scale=0.06
l=0.15
ax.arrow(l,0,0,numerical_fps[4][1]-arrow_scale,color=color)
ax.arrow(l,1,0,-(1-numerical_fps[4][1])+arrow_scale,color=color)
######### end arrows

######## xtick annotations ######
ax2.annotate('L',xy=(0.25,0.),xytext=(0.25,0.43),
             arrowprops=#dict(facecolor='black',width=linew,headwidth=1.2*linew,frac=0.2, shrink=.05)
             dict(arrowstyle="-", connectionstyle="arc3",linestyle="--",shrinkA=0.0,shrinkB=0.0),
            horizontalalignment='center',
            verticalalignment='bottom'
            )
ax2.annotate('H$_{0.75}$',xy=(0.75,0.),xytext=(0.75,0.36),
             arrowprops=#dict(facecolor='black',width=linew,headwidth=1.2*linew,frac=0.2, shrink=.05)
             dict(arrowstyle="-", connectionstyle="arc3",linestyle="--",shrinkA=0.0,shrinkB=0.0),
            horizontalalignment='center',
            verticalalignment='bottom'
            )
ax2.annotate('H$_{1.0}$',xy=(1.,0.),xytext=(1.,0.36),
             arrowprops=#dict(facecolor='black',width=linew,headwidth=1.2*linew,frac=0.2, shrink=.05)
             dict(arrowstyle="-", connectionstyle="arc3",linestyle="--",shrinkA=0.0,shrinkB=0.0),
            horizontalalignment='center',
            verticalalignment='bottom'
            )

ax.text(-.15,.91,'A',
        horizontalalignment='left',family="sans-serif",weight="heavy",
        transform=ax.transAxes)

ax2.text(-.15,.91,'B',
        horizontalalignment='left',family="sans-serif",weight="heavy",
        transform=ax2.transAxes)

plt.show()
    

