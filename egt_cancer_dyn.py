#!/usr/bin/env python3
#######################################################################
# egt_cancer_dyn.py
#Plot treatment dynamics for cycling between different treatment regimes.
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
import matplotlib.pyplot as plt
import matplotlib.ticker
import itertools
import plothelpers

####model parameters
mig=0.01 #spontanious phenotype switching rate
epsilon=e=0.04 #dormant cells growth rate
N0=1
lh=0.75 #0.75 #1.00 #0.75 #strong treatment strength
lw=0.25 #weak treatment strength
Tn=40 #duration before any treatment
segment_interval=20 #duration of treatment intervals
startfrequency_d=0.00 #proportion of dormant cells at start
surgical_reduction=(Tn,1e4) #(Surgery timepoint, number of cells left after surgical reduction)

Ts_steps=2. #length of strong treatment interval
Tw_steps=2. #length of weak treatment interval
total_segments=12 #total number of treatment cycles shown

Ts=segment_interval*Ts_steps
Tw=segment_interval*Tw_steps
Tmax=total_segments*segment_interval+Tn

####plot parameters
color_rp="#ECC000"
color_d="#D24E71"
color_number="#001889"
color_avfitness="black"

color_bg="black"
color_bg_n=(140/255.,140/255.,140/255.)
color_bg_h=(255/255.,255/255.,255/255.)
color_bg_l=(225/255.,225/255.,225/255.)
alpha_bg_n=1.#0.4
alpha_bg_h=1.#0.0
alpha_bg_l=1.#0.1
color_cycle=itertools.cycle([color_bg_h,color_bg_l])
alpha_cycle=itertools.cycle([alpha_bg_h,alpha_bg_l])

annot_cycle=itertools.cycle(["H","L"])
segment_seperator_lw=0.2

plotlabeloffset=+0.32
plotlabel_nocells_offset=-1.167*10**8
numberplot_maxn=1e10
plot_detailed_numbers=True #Also show number of D and RP cells, respecitvely

lw_data=1.8
lw_no_cells=lw_data
if plot_detailed_numbers:
    lw_no_cells=2*lw_data

tlim=250
tplotmin=10

with_axes="xy" #'xy' which of the axis ticks and labels to include
print_parameters=False #print the treatment parameters on top


ncolumns=1 #specifiy width of figure, one column or two columns
fig_height=None
if ncolumns==1:
    fig_height=2.5 #None
plothelpers.latexify(columns=ncolumns,fig_height=fig_height)
fig,ax=plt.subplots(3,1,sharex=True)
if ncolumns==1:
    fig.subplots_adjust(top=0.97, bottom=0.08, right=0.99, left=0.170,hspace=0.13)
elif print_parameters: 
    fig.subplots_adjust(top=0.94, bottom=0.07, right=0.98, left=0.100,hspace=0.11)
else:
    fig.subplots_adjust(top=0.98, bottom=0.08, right=0.98, left=0.100,hspace=0.11)
###########

### functions
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

M=set_mutation_matrix(mig)

def f(x,t,A,M):
    """ function defining the model dx/dt=f(x)"""
    x0=x[0]# if x[0] < 1 else 1.-1e-10
    x1=x[1]# if x[1] < 1 else 1.-1e-10
    phi=(-x.dot(A.dot(x)))
    x0_dot=M.dot(x*A.dot(x))[0]+x0*phi
    x1_dot=M.dot(x*A.dot(x))[1]+x1*phi
    return np.array([x0_dot,x1_dot])


def calc_av_fitness(x,A):
    return (x.dot(A.dot(x)))

def no_cells(av_fitness,dt=0.1, N0=1000,deathrate=0.,surgical_reduction=None):
    N=N0
    returnlist=[]
    if surgical_reduction is not None:
        t_s=surgical_reduction[0]
        surg_red=surgical_reduction[1]
    t=0
    for l in av_fitness:
        t=t+dt
        # l = l-np.mean(av_fitness)/2
        returnlist.append(N)
        dN=N * (np.exp(l*dt)-1)- N*deathrate
        # print (l,np.mean(av_fitness),N,dN)
        N=N+dN
        if surgical_reduction is not None and t>=t_s-0.001 and t<= t_s+0.001:
            N=surg_red
    return returnlist



def integrate(f,t_start=0,t_max=100,t_step=0.01,x0=np.array([1/2.,1/2.]),**kwargs):
    """ integrate dynamical system of function f """
    t=np.arange(t_start,t_max,t_step)
    sol=scipy.integrate.odeint(f,x0,t,rtol=1e-11,atol=1e-11,**kwargs) #1.49012e-8
    if (np.sum(sol[-1])>1) or np.any(sol[-1] < 0.):
        error=np.sum(sol[-1])
        # print("error in integration" ,sol[-1])
    return t,sol


#define parameter cycles for different treatment strenghts
payoff_matrices=itertools.cycle([set_payoff_array(e,lh),set_payoff_array(e,lw)])
durations_cycle=itertools.cycle([Ts,Tw])
durations_cycle,durations_cycle_copy=itertools.tee(durations_cycle)


sol=[]
av_fitness=[]
t=[]

####integrate without treatment
dur=Tn
payoff_matrix=set_payoff_array(e,0.)
def fun(x,t):
    return f(x,t,payoff_matrix,M)

time,solution=integrate(fun,x0=[startfrequency_d,1.0-startfrequency_d],t_max=dur)
av_f=np.array([calc_av_fitness(x,payoff_matrix) for x in solution])
av_fitness.append(av_f)
sol.append(solution)
t.append(time)

####integrate for the two alternating treatment regimes
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
cells=no_cells(av_fitness,dt=timestep,N0=N0,deathrate=0.0000,surgical_reduction=surgical_reduction)
cells_at_end_idx=np.searchsorted(tc,Tmax) #find number of cells at end
# print(cells[cells_at_end_idx]) #print the number of cells at Tmax



###############start plotting#############
#relative frequency
ax[0].plot(tc,sol[:,1],linewidth=lw_data,color=color_rp,label="RP")
ax[0].plot(tc,sol[:,0],linewidth=lw_data,color=color_d,label="D")

#average fitness
ax[1].plot(tc,av_fitness,linewidth=lw_data,color=color_avfitness,label="Average fitness")
ax[1].axhline(0.,linewidth=0.4,color="black",linestyle="-",label="zero fitness")

#number of cells
ax[2].plot(tc,cells,linewidth=lw_no_cells,color=color_number,label="all cells")
if plot_detailed_numbers:
    ax[2].plot(tc,sol[:,1]*cells,linewidth=lw_data,color=color_rp,label="RP")
    ax[2].plot(tc,sol[:,0]*cells,linewidth=lw_data,color=color_d,label="D")


for axi in ax:
    axi.margins(0.05)
    axi.set_xlim(xmax=Tmax,xmin=tplotmin)#tc[-1])
    axi.xaxis.set_major_locator(matplotlib.ticker.NullLocator())

# ax[0].set_ylim(ymin=-0.05,ymax=1.05)
ax[2].set_ylim(ymax=numberplot_maxn,ymin=1)
ax[2].set_yscale('log')

#### print background, annotations and sementation lines
# annot_text=next(annot_cycle) #correct starting point
dur=next(durations_cycle_copy) #correct starting point
last_whitebg_startingx=0 #for later label plotting
for times in t:
    if times[-1] < Tn: #first interval, no treatment
        fc=color_bg_n
        for axi in ax:
            axi.axvspan(times[0],times[-1],facecolor=fc,alpha=alpha_bg_n)
        continue

    dur=next(durations_cycle_copy)
    annot_text=next(annot_cycle)

    tsep=times[0]
    while tsep<times[-1]-0.5:
        ##plot line
        for axi in ax:
            axi.axvline(x=tsep,linestyle="-",color="black",linewidth=segment_seperator_lw)

        ##plot treatment strenght annotations
        trans= matplotlib.transforms.blended_transform_factory(
                ax[2].transData, ax[2].transAxes) # x in data untis, y in axes fraction
        def markerxpos(x,interval):
            return x+0.50*interval
        if markerxpos(tsep,segment_interval)>Tmax:
            break
        xpos=markerxpos(tsep,segment_interval)
        ypos=0.15
        ypos_data=trans.transform((xpos,ypos))[1]
        xpos_data=trans.transform((xpos,ypos))[0]
        if plothelpers.line_overlap(ax[2].get_lines()[0],xpos_data,ypos_data,ydiff=1e6):
            ypos=0.75
        ann = ax[2].annotate( xy=(xpos, ypos ),
                ha="center",xycoords=trans,s=annot_text)
        if annot_text=="H":
            last_whitebg_startingx=markerxpos(tsep,segment_interval)

        tsep+=segment_interval

    fc=next(color_cycle)
    alpha=next(alpha_cycle)
    for axi in ax:
        axi.axvspan(times[0],times[-1],facecolor=fc,alpha=alpha,lw=0.)


##### set axis labels
if not ncolumns==1:
    labelx=-0.075
else:
    labelx=-0.14
for axi in ax:
    axi.yaxis.set_label_coords(labelx, 0.5)

loc= matplotlib.ticker.MaxNLocator(6) # this locator puts ticks at regular intervals
ax[1].yaxis.set_major_locator(loc)
loc = matplotlib.ticker.LogLocator(numticks=5) # this locator puts ticks at log intervals
ax[2].yaxis.set_major_locator(loc)
ax[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

if 'x' in with_axes:
    ax[2].set_xlabel("Time")
if 'y' in with_axes:
    ax[0].set_ylabel("Rel. frequency")
    ax[2].set_ylabel("\# of cells")
    if ncolumns==2:
        ax[1].set_ylabel("Growth rate")
    else:
        ax[1].set_ylabel("Growth")

if not ncolumns == 1:
    ann = ax[2].annotate( xy=(Tn-11.5, 0.04 ), ha="center",xycoords=trans,s="no  \ntreat.")

##print number of cells at end
# trans = ax[2].transAxes # x and y in axes fraction units
# ax[2].text(0.88,0.25,"cells="+str(cells[cells_at_end]),
#    color=color_number,transform=trans,ha="center",va="center")

if (print_parameters==True):
    parameters=[mig,epsilon,lh,lw]
    parameter_names=["$\sigma","$\epsilon","$\lambda_H","$\lambda_L"]
    s=''
    for param,name in zip(parameters,parameter_names):
        s+=name+"="+str(param)+"$ "
    fig.suptitle(s,fontsize="large")

plothelpers.label_line(ax[0].get_lines()[0],last_whitebg_startingx+1.5,offset=+plotlabeloffset,alpha=0.)
plothelpers.label_line(ax[0].get_lines()[1],last_whitebg_startingx+1.5,offset=-plotlabeloffset,alpha=0.)
plothelpers.label_line(ax[2].get_lines()[0],last_whitebg_startingx,offset=+plotlabel_nocells_offset,alpha=0.)

plt.show()
plt.close()





