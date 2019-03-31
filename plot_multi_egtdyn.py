#!/usr/bin/env python3
#######################################################################
# plot_multi_egtdyn.py
#Plot treatment dynamics for cycling between different treatment regimes.
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

'''plots multiple figures with different parameters at once'''
import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
import itertools
import plothelpers
output_folder="./"
##define parameters for each plot
lhs=[1.,1.,1.,1.,1.,1.]
eps=[0.,0.1,0.,0.1,0.,0.1]
cycle_durations=[6,6,3,3,1,1]
plotlabel_corrections=[1.,1.,1.,1.,+2.9,-1.2]
outfilenames=[output_folder+str(i)+".eps" for i in range(len(lhs))]
labels=["y","","y","","xy","x"]
##


def plot_dynamics(cycle_duration=1,lh=0.75,e=0.1,mig=0.01,outfname="egtdyn.pdf",plotlabel_correction=1.,label='xy'):
    '''this creates essentially the same plot as "egt_cancer_dyn.py"'''
    ####model parameters
    mig_spont=mig #spontanious phenotype conversion rate
    mig_ind=0.00 #induced phenotype conversion rate
    epsilon=e #dormant cells growth rate
    base_growth=1. 
    comp_growth=1./2.
    N0=1
    lw=0.25 #weak treatment strength
    Tn=40 #duration before any treatment
    segment_interval=15 #duration of treatment intervals
    
    startfrequency_d=0.00
    
    surgical_reduction=(Tn,1e4) #Surgery timepoint, number of cells left after surgical reduction
    
    Ts_steps=cycle_duration
    Tw_steps=cycle_duration
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
    
    segment_seperator_lw=0.2
    annot_size=10
    
    plotlabeloffset=0.15*plotlabel_correction
    numberplot_maxn=1e11
    plot_detailed_numbers=True
    
    lw_data=1.8
    lw_no_cells=lw_data
    if plot_detailed_numbers:
        lw_no_cells=2*lw_data
    
    tlim=250
    tplotmin=10
    
    with_axes=label #can be 'xy', 'x', 'y' or None
    print_parameters=False
    
    annot_cycle=itertools.cycle(["H","L"])
    color_cycle=itertools.cycle([color_bg_h,color_bg_l])
    alpha_cycle=itertools.cycle([alpha_bg_h,alpha_bg_l])
    
    ### figure size -> essentially trial and error
    ncolumns=1 #specifiy width of figure, one column or two columns
    fig_height=None
    if ncolumns==1:
        fig_height=2.5 #None
    fig_height=2.4
    fig_width=2.4
    if 'y' in with_axes:
        fig_width=2.8
    if 'x' in with_axes:
        fig_height=2.5
    #plug in figure size
    plothelpers.latexify(columns=ncolumns,fig_height=fig_height,fig_width=fig_width)
    fig,ax=plt.subplots(3,1,sharex=True)
    ###adjust actual plot size inside figure depending on labels to plot
    if ncolumns==1:
        fig.subplots_adjust(top=0.97, bottom=0.08, right=0.99, left=0.170,hspace=0.13)
    elif print_parameters: 
        fig.subplots_adjust(top=0.94, bottom=0.09, right=0.98, left=0.250,hspace=0.11)
    else:
        fig.subplots_adjust(top=0.98, bottom=0.09, right=0.98, left=0.250,hspace=0.05)
    if 'x' not in with_axes:
        fig.subplots_adjust(bottom=0.03)
    if 'y' not in with_axes:
        fig.subplots_adjust( left=0.02)
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
    
    
        return np.array([x0_dot,x1_dot])
    
    def calc_av_fitness(x,l,mig_spont=mig_spont,mig_ind=mig_ind,epsilon=epsilon):
        """ function defining the model dx/dt=f(x)"""
    
    
        x0=x[0]# if x[0] < 1 else 1.-1e-10
        x1=x[1]# if x[1] < 1 else 1.-1e-10
        f0=epsilon
        f1=(base_growth-l-epsilon)*x0+(comp_growth-l)*x1
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
            # l = l-np.mean(av_fitness)/2
            returnlist.append(N)
            dN=N * (np.exp(l*dt)-1)- N*deathrate
            # print (l,np.mean(av_fitness),N,dN)
            N=N+dN
            if surgical_reduction is not None and t>=t_s-0.001 and t<= t_s+0.001:
                N=surg_red
    
    
        return returnlist
    
    
    
    def integrate(f,t_start=0,t_max=100,t_step=0.01,x0=np.array([1/2.,1/2.]),**kwargs):
        t=np.arange(t_start,t_max,t_step)
        sol=scipy.integrate.odeint(f,x0,t,rtol=1e-11,atol=1e-11,**kwargs) #1.49012e-8
        if (np.sum(sol[-1])>1) or np.any(sol[-1] < 0.):
            error=np.sum(sol[-1])
            # print("error in integration" ,sol[-1])
        return t,sol
    
    
    
    
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
        return f(x,t,l)
    
    time,solution=integrate(fun,x0=[startfrequency_d,1.0-startfrequency_d],t_max=dur)
    # print(solution)
    av_f=np.array([calc_av_fitness(x,l) for x in solution])
    av_fitness.append(av_f)
    sol.append(solution)
    t.append(time)
    
    # av_fitness=[]
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
    
    # print(fun(solution[-1],0.))
    sol=np.concatenate(sol)
    av_fitness=np.concatenate(av_fitness)
    tc=np.concatenate(t)
    timestep=(tc[-1]-tc[-2])
    cells=no_cells(av_fitness,dt=timestep,N0=N0,deathrate=0.0000,surgical_reduction=surgical_reduction)
    cells_at_end=np.searchsorted(tc,Tmax) #find number of cells at end
    # print(cells[cells_at_end])
    
    
    
    ###############start plotting#############
    ax[0].plot(tc,sol[:,1],linewidth=lw_data,color=color_rp,label="P")
    ax[0].plot(tc,sol[:,0],linewidth=lw_data,color=color_d,label="D")
    
    ax[1].plot(tc,av_fitness,linewidth=lw_data,color=color_avfitness,label="Average fitness")
    ax[1].axhline(0.,linewidth=0.4,color="black",linestyle="-",label="zero fitness")
    
    ax[2].plot(tc,cells,linewidth=lw_no_cells,color=color_number,label="all cells")
    if plot_detailed_numbers:
        ax[2].plot(tc,sol[:,1]*cells,linewidth=lw_data,color=color_rp,label="RP")
        ax[2].plot(tc,sol[:,0]*cells,linewidth=lw_data,color=color_d,label="D")
    
    for axi in ax:
        axi.margins(0.05)
        axi.set_xlim(xmax=Tmax,xmin=tplotmin)#tc[-1])
        axi.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    
    # ax[0].set_ylim(ymin=-0.05,ymax=1.05)
    ax[1].set_ylim(ymin=-0.55,ymax=0.55)
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
            # print(tsep,times[-1])
            for axi in ax:
                axi.axvline(x=tsep,linestyle="-",color="black",linewidth=segment_seperator_lw)
    
            ##plot annotation
            trans = ax[2].get_xaxis_transform() # x in data untis, y in axes fraction
            def markerxpos(x,interval):
                return x+0.50*interval
            if markerxpos(tsep,segment_interval)>Tmax:
                break

            xpos=markerxpos(tsep,segment_interval)
            ypos=0.15
            ypos_data=trans.transform((xpos,ypos))[1]
            xpos_data=trans.transform((xpos,ypos))[0]
            # inv = trans.inverted()
            # point=trans.transform((xpos,ypos))
            # print(point,(xpos,ypos),inv.transform(point))
            if plothelpers.line_overlap(ax[2].get_lines()[2],xpos,ypos_data,ydiff=1e5):
                ypos=0.75
            ann = ax[2].annotate( xy=(xpos,ypos), ha="center",xycoords=trans,s=annot_text,size=annot_size)
            if annot_text=="H":
                last_whitebg_startingx=markerxpos(tsep,segment_interval)
    
            tsep+=segment_interval
    
        fc=next(color_cycle)
        alpha=next(alpha_cycle)
        for axi in ax:
            axi.axvspan(times[0],times[-1],facecolor=fc,alpha=alpha)
    
    
    
    
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
        ax[0].set_ylabel("Pop. fraction")
        ax[2].set_ylabel("\# of cells")
        if ncolumns==2:
            ax[1].set_ylabel("Growth rate")
        else:
            ax[1].set_ylabel("Growth")
    else:
        for axe in ax:
            axe.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    
    if not ncolumns == 1:
        ann = ax[2].annotate( xy=(Tn-11.5, 0.04 ), ha="center",xycoords=trans,s="no  \ntreat.")
    
    if (print_parameters==True):
        parameters=[mig,epsilon,lh,lw]
        parameter_names=["$\sigma","$\epsilon","$\lambda_H","$\lambda_L"]
        s=''
        for param,name in zip(parameters,parameter_names):
            s+=name+"="+str(param)+"$ "
        # fig.text(0.5,0.03,s,color="black",ha="center",va="center")
        fig.suptitle(s,fontsize="large")
    
    plothelpers.label_line(ax[0].get_lines()[0],last_whitebg_startingx,offset=+plotlabeloffset,alpha=0.)
    plothelpers.label_line(ax[0].get_lines()[1],last_whitebg_startingx,offset=-plotlabeloffset,alpha=0.)
    # plothelpers.label_line(ax[2].get_lines()[0],last_whitebg_startingx,offset=+10**8,alpha=0.)
    
    # plt.savefig(outfname)
    plt.show()
    plt.close()

#do the actual plotting
for  lh,cycle,corr,outfilename,label,e in zip(lhs,cycle_durations,plotlabel_corrections,outfilenames,labels,eps):
    plot_dynamics(lh=lh,cycle_duration=cycle,plotlabel_correction=corr,e=e,outfname=outfilename,label=label)



