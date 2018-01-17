import matplotlib
import matplotlib.pyplot
import itertools
import numpy as np
import math
def latexify(fig=None,fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 2.825 if columns==1 else 5.788 # width in inches
        # fig_width = 3.38 if columns==1 else 7. # width in inches
        # fig_width = 3.176 if columns==1 else 6.491 # width in inches
        # fig_width = 3.39 if columns==1 else 6.9 # width in inches
        # 1 inch= 2.54 cm

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES


    params = {#'backend': 'ps',
              # 'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 9, # fontsize for x and y labels (was 10)
              'axes.titlesize': 9,
              'font.size': 10, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'sans-serif',
              'font.sans-serif': ['computer modern roman'], #avoid bold axis label
              'text.latex.preamble': [r'\usepackage{helvet}',    # set the normal font here
                                r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                                r'\sansmath'  ]             # <- tricky! -- gotta actually tell tex to use!
    }
    if fig:
        print("texify figure dimensions set: ",fig_width,fig_height)
        fig.set_size_inches((fig_width,fig_height),forward=True)
    matplotlib.rcParams.update(params)
    return params


def create_colorcyle(number,cmap=None,cmapname="viridis"):
    if not cmap:
        cmap = matplotlib.pyplot.get_cmap(cmapname)
    indices = np.linspace(0, cmap.N, number)
    my_colors = itertools.cycle([cmap(int(i)) for i in indices])
    return my_colors

def create_markercycle(number):
    markerpref=["o","v","s","D","^","<",">","+","x",".","X","1","2","3","4","8"]
    if number <= len (markerpref):
        markers=markerpref
    else:
        markers = [(int(2+i/2), 1+i%2, 0) for i in range(number)]
    return itertools.cycle(markers)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def resadjust(ax, xres=None, yres=None):
    """
    Send in an axis and I fix the resolution as desired.
    """

    if xres:
        start, stop = ax.get_xlim()
        ticks = np.arange(start, stop + xres, xres)
        ax.set_xticks(ticks)
    if yres:
        start, stop = ax.get_ylim()
        ticks = np.arange(start, stop + yres, yres)
        ax.set_yticks(ticks)



def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def align_labels(axes_list,axis='y',align=None):
    if align is None:
        align = 'l' if axis == 'y' else 'b'
    yx,xy = [],[]
    for ax in axes_list:
        yx.append(ax.yaxis.label.get_position()[0])
        xy.append(ax.xaxis.label.get_position()[1])

    if axis == 'x':
        if align in ('t','top'):
            lim = max(xy)
        elif align in ('b','bottom'):
            lim = min(xy)
    else:
        if align in ('l','left'):
            lim = min(yx)
        elif align in ('r','right'):
            lim = max(yx)

    if align in ('t','b','top','bottom'):
        for ax in axes_list:
            t = ax.xaxis.label.get_transform()
            x,y = ax.xaxis.label.get_position()
            ax.xaxis.set_label_coords(x,lim,t)
    else:
        for ax in axes_list:
            t = ax.yaxis.label.get_transform()
            x,y = ax.yaxis.label.get_position()
            ax.yaxis.set_label_coords(lim,y,t)


#Label line with line2D label data
#from http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
def label_line(line,x,offset=None,label=None,align=False,alpha=None,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # # Convert datetime objects to floats
    # define datetime first (from datetime import datetime)
    # if isinstance(x, datetime):
    #     x = matplotlib.dates.date2num(x)

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = math.degrees(math.atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    if offset is None:
        offset=0.
    y = y + offset

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor() #deprecated: get_axis_bgcolor()

    if 'alpha' is None:
        alpha=1.


    # if 'clip_on' not in kwargs:
    #     kwargs['clip_on'] = True

    # if 'zorder' not in kwargs:
    #     kwargs['zorder'] = 2.5

    t=ax.text(x,y,label,rotation=trans_angle,**kwargs)
    t.set_bbox(dict( alpha=alpha,facecolor=kwargs['backgroundcolor']))

def label_lines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)


def line_overlap(line,x,y,ydiff=None):
    if ydiff is None:
        ydiff=y
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # # Convert datetime objects to floats
    # define datetime first (from datetime import datetime)
    # if isinstance(x, datetime):
    #     x = matplotlib.dates.date2num(x)

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return
    #Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    yline = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    # print(x,y,ydiff,xdata[ip],yline)

    if y<yline+ydiff and y > yline-ydiff:
        return True
    else:
        return False
