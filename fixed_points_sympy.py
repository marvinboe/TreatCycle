#!/usr/bin/env python3
#######################################################################
# fixed_points_sympy.py
#Plot bifurcation diagram for treatment strength in EGT model 
#of Glioblastoma.
#
#Copyright 2018 Marvin A. Böttcher
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

'''Stable fixed points are calculated and saved into "fps.txt" if the 
file doesn't exist. If it the file exists, the data from the file 
is used for plotting and no new fixed points are calculated.'''

import sympy
import plothelpers
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

sympy.init_printing()


ms=0.1 #phenotype migration rate
mi=0.0 #phenotype migration rate

eps=0.1 #dormant cells growth rate

lmax=1.3 
ls=np.linspace(0.0,lmax,20) #treatment strength lambda values to test

thresh=1e-19 #threshold for stability of FP (0 doesn't work for some reason, numerical issues?!)

def f0(x0,x1,l,ms,mi,eps):
    '''Growth of dormant cells'''
    f0=eps
    f1=(1-eps-l)*x0+(0.5-l)*x1
    fbar=f0*x0+f1*x1
    x0_dot=sympy.Piecewise(
            ((f0-fbar)*x0 -mi*fbar*x0 +ms*(x1-x0),fbar>=0),
            ((f0-fbar)*x0 -mi*fbar*x1 +ms*(x1-x0),True))
    return x0_dot

def f1(x0,x1,l,ms,mi,eps):
    '''Growth of raplidly proliferating cells'''
    f0=eps
    f1=(1-eps-l)*x0+(0.5-l)*x1
    fbar=f0*x0+f1*x1
    x1_dot=sympy.Piecewise(
            ((f1-fbar)*x1 +mi*fbar*x0 -ms*(x1-x0),fbar>0),
            ((f1-fbar)*x1 +mi*fbar*x1 -ms*(x1-x0),True))
    # if l<=0.5:
    #     x1_dot=(f1-fbar)*x1 +mi*fbar*x0 -ms*(x1-x0)
    # else:
    #     x1_dot=(f1-fbar)*x1 -mi*fbar*x0 -ms*(x1-x0)
    return x1_dot

def f(x0,x1,l,mig_spont,mig_ind,eps):
    """ function defining the model dx/dt=f(x)"""
    return [f0(x0,x1,l,mig_spont,mig_ind,eps),f1(x0,x1,l,mig_spont,mig_ind,eps)]


def fps_from_analytics(lamb,eps,ms):
    '''analytical solution of fixed points, calculated previously with Sympy'''
    e_d = [-2*lamb/(3*(2*eps - 1)) - (4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))/(3*(8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)) - (8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)/3, -2*lamb/(3*(2*eps - 1)) - (4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))/(3*(sympy.Rational(-1, 2) - sympy.sqrt(3)*sympy.I/2)*(8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)) - (sympy.Rational(-1, 2) - sympy.sqrt(3)*sympy.I/2)*(8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)/3, -2*lamb/(3*(2*eps - 1)) - (4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))/(3*(sympy.Rational(-1, 2) + sympy.sqrt(3)*sympy.I/2)*(8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)) - (sympy.Rational(-1, 2) + sympy.sqrt(3)*sympy.I/2)*(8*lamb**3/(2*eps - 1)**3 - 9*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 27*ms/(2*eps - 1) + sympy.sqrt(-4*(4*lamb**2/(2*eps - 1)**2 - 3*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1))**3 + (16*lamb**3/(2*eps - 1)**3 - 18*lamb*(-2*eps - 2*lamb + 4*ms + 1)/(2*eps - 1)**2 - 54*ms/(2*eps - 1))**2)/2)**sympy.Rational(1, 3)/3]
    return e_d


def av_fitness(x0,l,ms,mi,eps):
    '''average fitness dependent on parameters and population composition'''
    if x0 is None:
        return None
    x1=1-x0
    f0=eps
    f1=(1-eps-l)*x0+(0.5-l)*x1
    fbar=f0*x0+f1*x1
    return fbar
av_fitness=np.vectorize(av_fitness)


    
def find_stable_fp(l,ms,mi,eps):
    '''returns stable fixed points for parameters. If no stable 
    fixed points can be found (numerical error) returns unique 
    saddle point and prints warning.'''
    #use analytical result without phenotype conversion
    if ms <= 0 and mi <=0 :
        if l < 1/2-eps:
            return [0,1]
        elif l< 1-2*eps:
            temp=2*l/(1-2*eps) -1
            return [temp,1-temp]
        else:
            return [1,0]

    x0,x1=sympy.symbols('x0 x1')
    fps=sympy.solve([f0(x0,x1,l,ms,mi,eps),
        sympy.Eq(x0+x1-1.)],x0,x1)
    fps_alt=fps_from_analytics(l,eps,ms)
    if len(fps)==0: #catch numerical errors
        raw=np.array([sympy.N(sympy.functions.re(x)) for x in fps_alt])
        fps=np.array([raw,1-raw]).T
        print("using alternative fps:",[sympy.N(sympy.functions.re(x)) for x in fps_alt],[sympy.N(sympy.functions.im(x)) for x in fps_alt])
        print()

    #check FPs
    saddle_list=[]
    for fp in fps:
        #calculate EVs of jacobian for system
        F=sympy.Matrix([f0(x0,x1,l,ms,mi,eps),f1(x0,x1,l,ms,mi,eps)])
        jac=F.jacobian([x0,x1])
        j_fp=jac.subs([(x0,fp[0]),(x1,fp[1])])
        evs=j_fp.eigenvals().keys()

        #stable if both FP < 0 --- numerical errors!
        stable=[sympy.re(ev)<=thresh for ev in evs]
        print([float(sympy.re(f)) for f in fp],stable,[float(sympy.re(ev)) for ev in evs])
        print()

        if np.all(stable):
            return fp
        elif np.any(stable):
            saddle_list.append(fp)


    #check limits:
    for fp in [[0,1],[1.,0.]]:
        F=sympy.Matrix([f0(x0,x1,l,ms,mi,eps),f1(x0,x1,l,ms,mi,eps)])
        jac=F.jacobian([x0,x1])
        j_fp=jac.subs([(x0,fp[0]),(x1,fp[1])])
        evs=j_fp.eigenvals().keys()
        stable=[sympy.re(ev)<0. for ev in evs]
        print("limits",[float(sympy.re(f)) for f in fp],stable,[float(sympy.re(ev)) for ev in evs])

        if np.all(stable):
            return fp


    #numerical errors?!
    if len(saddle_list)==1:
        print("no stable FP found, but found saddle... ",saddle_list[0])
        return saddle_list[0]
    elif len(saddle_list) > 1:
        print("no stable FP found, but found several saddles ",saddle_list)
    else:
        print("no stable FP, no saddles found!!! ",fps)
        
    return None

def return_fplist(ls,ms,mi,eps):
    '''Calculates list of stable FP for list of treatment
    strengths lambda ls.'''
    fps=[]
    for l in ls:
        fp=find_stable_fp(l,ms,mi,eps)
        
        if fp is not None:
            fps.append(sympy.re(fp[0]))
        else :
            fps.append(-1.)
    return fps



def plot_fps(ax,ax2,fps,color=None,ls=ls,ms=ms,mi=mi,eps=eps,label=" ",linestyle='-'):
    '''plot FPs (fps) nicely'''
    linew=1.8
    #plot phenotpye composition
    ax.plot(ls,fps,color=color,lw=linew,label=label,ls=linestyle,alpha=0.95)
    #plot growth rate (no phenotype conversion)
    ax2.plot(ls,av_fitness(fps,ls,0,0,eps),lw=linew,color=color,ls=linestyle,label=label,alpha=0.95)#,label="av. fitness")
    
    ############legend hack#########
    #change color
    # color=next(colors) #cmap(epsilon/np.max(e)/2.)
    # ax.plot(np.nan,".",marker='.',color=color,markersize=msize,mew=linew,label="$\sigma="+str(0.)+"$")
    # ax2.plot(np.nan,".",marker='.',color=color,markersize=msize,mew=linew,label="$\sigma="+str(ms+mi)+"$")
    ############end legend hack#####

##load old data, otherwise generate new    
try:
    fps=np.loadtxt("fps.txt")
    if fps is None:
        os.remove("fps.txt")
        print("file 'fps.txt' empty or invalid -> deleted. Please restart script.")
        exit(0)
    
except:
    fps_noswitch=np.array(return_fplist(ls,0,0,eps))
    fps_switch=np.array(return_fplist(ls,ms,mi,eps))
    fps_zero_eps=np.array(return_fplist(ls,ms,mi,0.))
    fps_noswitch_zero_eps=np.array(return_fplist(ls,0.,0.,0.))
    fps=np.array([ls,fps_noswitch,fps_switch,fps_zero_eps,fps_noswitch_zero_eps]).T
    np.savetxt("fps.txt",fps)
    
print(fps)

#initializes plots
plothelpers.latexify(columns=2,fig_height=2.3)
fig,axes=plt.subplots(1,2,sharex=True)#,figsize=(3.39,2.8)
ax=axes[0]
ax2=axes[1]
ax.margins(0.06)
ax2.margins(0.06)

# colors=plothelpers.create_colorcyle(4,cmapname="tab20b")#creates itertools.cycle with four different colors
colors=itertools.cycle(['blue','blue','orange','orange'])
linestyles=itertools.cycle(['-','--'])

#plot stable FP (no phenotype conversion)
color=next(colors)
linestyle=next(linestyles)
label="$\epsilon=0.1,\sigma=0$"
plot_fps(ax,ax2,fps[:,1],ms=0,mi=0,color=color,linestyle=linestyle,label=label)


#plot stable FP with phenotype conversion
color=next(colors)
linestyle=next(linestyles)
label="$\epsilon=0.1$"+"$,\sigma="+str(ms)+"$" #\sigma_i="+str(mi)+",
plot_fps(ax,ax2,fps[:,2],color=color,linestyle=linestyle,label=label)

#plot stable FP with zero D growth and no conversion
color=next(colors)
linestyle=next(linestyles)
label="$\epsilon=0.0,\sigma=0$" #\sigma_i="+str(mi)+",
plot_fps(ax,ax2,fps[:,4],eps=0.,ms=0.,color=color,linestyle=linestyle,label=label)

#plot stable FP with zero D growth
color=next(colors)
linestyle=next(linestyles)
label="$\epsilon=0.0$"+"$,\sigma="+str(ms)+"$" #\sigma_i="+str(mi)+",
plot_fps(ax,ax2,fps[:,3],eps=0.,color=color,linestyle=linestyle,label=label)


# #######draw arrows pointing in direction of dynamics################
# headw=0.03
# headl=0.05
#
# #middle arrow (from bottom and top)
# color=next(colors)
# offset_edge=0.01
# offset_graph=0.10
# pos=25
# ax.arrow(fps[pos,0],offset_edge,0,fps[pos,1]-(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
# ax.arrow(fps[pos,0],1-offset_edge,0,-(1-fps[pos,1])+(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
#
# #from bottom and top
# color=next(colors)
# pos=12
# offset_edge=0.02
# offset_graph=0.10
# ax.arrow(fps[pos,0],offset_edge,0,fps[pos,2]-(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
# ax.arrow(fps[pos,0],1-offset_edge,0,-(1-fps[pos,2])+(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
#
# #from bottom and top
# color=next(colors)
# pos=37
# offset_edge=0.02
# offset_graph=0.10
# ax.arrow(fps[pos,0],offset_edge,0,fps[pos,3]-(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
# ax.arrow(fps[pos,0],1-offset_edge,0,-(1-fps[pos,3])+(offset_edge+offset_graph),color=color,head_width=headw, head_length=headl)
# ######### end arrows


######## xtick annotations ######
ax2.annotate('$\lambda_L$',xy=(0.25,-0.08),xytext=(0.25,0.43),
             arrowprops=#dict(facecolor='black',width=linew,headwidth=1.2*linew,frac=0.2, shrink=.05)
             dict(arrowstyle="-", connectionstyle="arc3",linestyle="--",shrinkA=0.0,shrinkB=0.0),
            horizontalalignment='center',
            verticalalignment='bottom'
            )
ax2.annotate('$\lambda_H$',xy=(1.0,-0.08),xytext=(1.0,0.43), #$_{0.75}$
             arrowprops=#dict(facecolor='black',width=linew,headwidth=1.2*linew,frac=0.2, shrink=.05)
             dict(arrowstyle="-", connectionstyle="arc3",linestyle="--",shrinkA=0.0,shrinkB=0.0),
            horizontalalignment='center',
            verticalalignment='bottom'
            )

ax.text(-.18,.91,'A',
        horizontalalignment='left',family="sans-serif",weight="heavy",
        transform=ax.transAxes)

ax2.text(-.18,.91,'B',
        horizontalalignment='left',family="sans-serif",weight="heavy",
        transform=ax2.transAxes)

ax.set_xlabel("Treatment cost $\lambda$")
ax2.set_xlabel("Treatment cost $\lambda$")
ax.set_ylabel("D cell fraction")
ax2.set_ylabel("Average growth")
leg=ax.legend(loc=2)

fig.tight_layout(pad=0.4)
# fig.subplots_adjust(top=0.99, bottom=0.16, right=0.99, left=0.08,wspace=0.22)

plt.show()

