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
mig_spont=0.010 #phenotype conversion rate
mig_ind=0.0 #induced phenotype conversion rate 
base_growth=1
comp_growth=1/2.

epsilon=e=0.0 #dormant cells growth rate
N0=1
lh=1.0 #1.00 #0.75 #strong treatment strength
lw=0.25#weak treatment strength
Tn=40 #duration before any treatment
segment_interval=15 #duration of treatment intervals
startfrequency_d=0.10 #proportion of dormant cells at start
surgical_reduction=(Tn,1e3) #None #(Surgery timepoint, number of cells left after surgical reduction)

Ts_steps=2. #length of strong treatment interval
Tw_steps=2. #length of weak treatment interval
total_segments=12 #total number of treatment cycles shown

Ts=segment_interval*Ts_steps
Tw=segment_interval*Tw_steps
Tmax=total_segments*segment_interval+Tn

####plot parameters
plotlabeloffset=+0.16 #position of plotlabels at lines

color_rp="#ECC000"
color_d="#D24E71"
color_number="#001889"
color_avfitness="black"

color_bg="black"
color_bg_n=(140/255.,140/255.,140/255.)
color_bg_h=(255/255.,255/255.,255/255.)
color_bg_l=(225/255.,225/255.,225/255.)
alpha_bg_n=1.
alpha_bg_h=1.
alpha_bg_l=1.
color_cycle=itertools.cycle([color_bg_h,color_bg_l])
alpha_cycle=itertools.cycle([alpha_bg_h,alpha_bg_l])

annot_cycle=itertools.cycle(["H","L"])
segment_seperator_lw=0.2

plotlabel_nocells_offset=2.*10**8
numberplot_maxn=1e10
plot_detailed_numbers=True #Also show number of D and P cells, respecitvely

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
#     fig.subplots_adjust(top=0.97, bottom=0.08, right=0.99, left=0.170,hspace=0.13)
# elif print_parameters: 
#     fig.subplots_adjust(top=0.94, bottom=0.07, right=0.98, left=0.100,hspace=0.11)
# else:
#     fig.subplots_adjust(top=0.98, bottom=0.08, right=0.98, left=0.100,hspace=0.11)
plothelpers.latexify(columns=ncolumns,fig_height=fig_height)
fig,ax=plt.subplots(3,1,sharex=True)
###########

### functions

def f(x,t,l,mig_spont=mig_spont,mig_ind=mig_ind,epsilon=epsilon):
    """ function defining the model dx/dt=f(x)"""

    x0=x[0]# if x[0] < 1 else 1.-1e-10
    x1=x[1]# if x[1] < 1 else 1.-1e-10
    f0=epsilon
    f1=((base_growth-l-epsilon)*x0+(comp_growth-l)*x1)#*x0+(1/2.-l)*x1
    fbar=f0*x0+f1*x1

    if fbar > 0:# True: #
        x0_dot=(f0-fbar)*x0 -mig_ind*fbar*x0 +mig_spont*(x1-x0)
        x1_dot=(f1-fbar)*x1 +mig_ind*fbar*x0 -mig_spont*(x1-x0)
    else:
        x0_dot=(f0-fbar)*x0 - mig_ind*fbar*x1 +mig_spont*(x1-x0)
        x1_dot=(f1-fbar)*x1 + mig_ind*fbar*x1 -mig_spont*(x1-x0)
    if x0 < -0.0001 or x1 < -0.0001 :
        print("error",x0,x1,x0_dot,x1_dot)
        exit(0)
    if x0_dot+x0>0 and x1_dot+x1 > 0 : 
        return np.array([x0_dot,x1_dot])
    elif x0_dot+x0<0:
        return np.array([x0,x1_dot])
    elif x1_dot+x1<0:
        return np.array([x0_dot,x1])

def calc_av_fitness(x,l,mig_spont=mig_spont,mig_ind=mig_ind,epsilon=epsilon):
    """ function returning average fitness for the model """

    x0=x[0]# if x[0] < 1 else 1.-1e-10
    x1=x[1]# if x[1] < 1 else 1.-1e-10
    f0=epsilon
    f1=(1.-l-epsilon)*x0+(1/2.-l)*x1
    fbar=f0*x0+f1*x1
    return fbar


def no_cells(av_fitness,dt=0.1, N0=1000,deathrate=0.,surgical_reduction=None):
    N=N0
    returnlist=[]
    if surgical_reduction is not None:
        t_s=surgical_reduction[0]
        surg_red=surgical_reduction[1]
    t=0
    for l in av_fitness:
        t=t+dt
        returnlist.append(N)
        dN=N * (np.exp(l*dt)-1)- N*deathrate
        N=N+dN
        if surgical_reduction is not None and t>=t_s-0.001 and t<= t_s+0.001:
            N=surg_red
    return returnlist



def integrate(f,t_start=0,t_max=100,t_step=0.01,x0=np.array([1/2.,1/2.]),**kwargs):
    """ integrate dynamical system of function f """
    t=np.arange(t_start,t_max,t_step)
    sol=scipy.integrate.odeint(f,x0,t,rtol=1e-11,atol=1e-11,**kwargs) #1.49012e-8
    if (np.sum(sol[1])>1) or np.any(sol[-1] < 0.) or np.any(sol) > 1.01:
        error=np.sum(sol[-1])
        print("error in integration" ,sol[-1])
    return t,sol


#define parameter cycles for different treatment strenghts
treat_strengths=itertools.cycle([lh,lw])
durations_cycle=itertools.cycle([Ts,Tw])
durations_cycle,durations_cycle_copy=itertools.tee(durations_cycle)

sol=[]
av_fitness=[]
t=[]

####integrate without treatment
dur=Tn
l=0.
def fun(x,t):
    return f(x,t,l,mig_spont,mig_ind)

time,solution=integrate(fun,x0=[startfrequency_d,1.0-startfrequency_d],t_max=dur)
av_f=np.array([calc_av_fitness(x,l) for x in solution])
av_fitness.append(av_f)
sol.append(solution)
t.append(time)

#start with low dosage: uncomment following
# dur=next(durations_cycle)
# l=next(treat_strengths)

####integrate for the two alternating treatment regimes
while (time[-1]<Tmax):
    dur=next(durations_cycle)
    l=next(treat_strengths)

    def fun(x,t):
        return f(x,t,l)

    if (abs(solution[-1].any()) > 1.):
        print ("error")
        break
    else:
        time,solution=integrate(fun,x0=solution[-1],t_max=time[-1]+dur,t_start=time[-1],mxstep=5000000)
    av_f=np.array([calc_av_fitness(x,l) for x in solution])
    av_fitness.append(av_f)
    sol.append(solution)
    t.append(time)

#concatenates solutions
sol=np.concatenate(sol)
av_fitness=np.concatenate(av_fitness)
tc=np.concatenate(t)
timestep=(tc[-1]-tc[-2])
cells=no_cells(av_fitness,dt=timestep,N0=N0,deathrate=0.0000,surgical_reduction=surgical_reduction)
cells_at_end_idx=np.searchsorted(tc,Tmax) #find number of cells at end
# print(cells[cells_at_end_idx]) #print the number of cells at Tmax
print(sol[-1],av_fitness[-1])



###############starts plotting#############
#relative frequency
ax[0].plot(tc,sol[:,1],linewidth=lw_data,color=color_rp,label="P")
ax[0].plot(tc,sol[:,0],linewidth=lw_data,color=color_d,label="D")

#average fitness
ax[1].plot(tc,av_fitness,linewidth=lw_data,color=color_avfitness,label="Average fitness")
ax[1].axhline(0.,linewidth=0.4,color="black",linestyle="-",label="zero fitness")

#number of cells
ax[2].plot(tc,cells,linewidth=lw_no_cells,color=color_number,label="all cells")
if plot_detailed_numbers:
    ax[2].plot(tc,sol[:,1]*cells,linewidth=lw_data,color=color_rp,label="P")
    ax[2].plot(tc,sol[:,0]*cells,linewidth=lw_data,color=color_d,label="D")


for axi in ax:
    axi.margins(0.05)
    axi.set_xlim(xmax=Tmax,xmin=tplotmin)#tc[-1])
    axi.xaxis.set_major_locator(matplotlib.ticker.NullLocator())

# ax[0].set_ylim(ymin=-0.05,ymax=1.05)
ax[2].set_ylim(ymax=numberplot_maxn,ymin=1)
ax[2].set_yscale('log')

#### print background, annotations and sementation lines
annot_text=next(annot_cycle) #correct starting point (comment out for LD first)
dur=next(durations_cycle_copy) #correct starting point (comment out for LD first)
annot_text=next(annot_cycle)
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
        axis_to_data = ax[2].transAxes + ax[2].transData.inverted()
        ypos_data= axis_to_data.transform((xpos,ypos))[1]
        xpos_data=xpos 
        if plothelpers.line_overlap(ax[2].get_lines()[0],xpos_data,ypos_data,ydiff=1e5):
            ypos=0.75
            ypos_data= axis_to_data.transform((xpos,ypos))[1]
        ann = ax[2].annotate( xy=(xpos_data, ypos_data ),
                ha="center",s=annot_text) #xycoords=trans,
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


# ax[0].set_ylim(ymin=0,ymax=1)
ax[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))# this locator puts ticks at regular intervals
ax[2].yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))# this locator puts ticks at log intervals
ax[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

if 'x' in with_axes:
    ax[2].set_xlabel("Time")
if 'y' in with_axes:
    ax[0].set_ylabel("Pop. fraction")
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
##

##print parameters in title of plot
if (print_parameters==True):
    parameters=[epsilon,mig_spont,lh,lw]
    parameter_names=["$\epsilon","$\sigma","$\lambda_H","$\lambda_L"]
    s=''
    for param,name in zip(parameters,parameter_names):
        s+=name+"="+str(param)+"$ "
    fig.suptitle(s,fontsize="small")
##

##label lines directly
plothelpers.label_line(ax[0].get_lines()[0],last_whitebg_startingx+1.5,offset=+plotlabeloffset,alpha=0.)
plothelpers.label_line(ax[0].get_lines()[1],last_whitebg_startingx+1.5,offset=-plotlabeloffset,alpha=0.)
# plothelpers.label_line(ax[2].get_lines()[0],last_whitebg_startingx,offset=+plotlabel_nocells_offset,alpha=0.)
##

fig.tight_layout(pad=0.8)
fig.subplots_adjust(hspace=0.1,bottom=0.07,right=0.96)
for a in ax:
    a.yaxis.set_tick_params(pad=1)
plt.show()
plt.close()





