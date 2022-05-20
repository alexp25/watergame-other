import matplotlib.pylab as plt
import matplotlib
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from typing import List
import numpy as np
import math
import copy
from colour import Color

import matplotlib as mpl
from svgpath2mpl import parse_path
from xml.dom import minidom
from matplotlib.ticker import MaxNLocator


# FSIZE_TITLE = 16
# FSIZE_LABEL = 14
# FSIZE_LABEL_S = 14
# FSIZE_LABEL_XS = 12


FSIZE_TITLE = 16
FSIZE_LABEL = 14
FSIZE_LABEL_S = 14
FSIZE_LABEL_XS = 12
OPACITY = 0.9
OPACITY = 1


class Timeseries:
    # declare props as object NOT class props!
    def __init__(self):
        self.label = "None"
        self.color = "blue"
        self.x = []
        self.y = []


class Barseries:
    # declare props as object NOT class props!
    def __init__(self):
        self.label = "None"
        self.color = "blue"
        self.data = []
        self.average = None


class CMapMatrixElement:
    def __init__(self):
        self.i = 0
        self.j = 0
        self.ilabel = ""
        self.jlabel = ""
        self.val = 0
        self.auxval = 0

# class Timeseries(object):
#     label = "None"
#     color = "blue"
#     x = []
#     y = []


def plot(y, x, title, xlabel, ylabel, show):
    fig = plt.figure(figsize=(8,6))
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    yint = []
    plt.grid()
    locs, labels = plt.xticks()
    yint = x
    # for each in locs:
    #     yint.append(int(each))
    plt.xticks(yint)
    set_disp(title, xlabel, ylabel)
    if show:
        plt.show()
    return fig


def plot_xy(x, y, rads, labels, colors, title, xlabel, ylabel, scale, show_legend, labels_points):
    fig, ax = plt.subplots()
    # figsize=(8, 6)

    set_plot_font(FSIZE_LABEL_XS)

    if not scale:
        pass
    else:
        ax.set_xlim(scale[0])
        ax.set_ylim(scale[1])

    ax.grid(zorder=0)

    scatter_vect = []

    for i in range(len(x)):
        scatter_vect.append(ax.scatter(
            y[i], x[i], s=rads[i], c=colors[i], alpha=OPACITY, zorder=3))

    for i, txt in enumerate(labels_points):
        col = Color(colors[i][:-2])
        if col.luminance < 0.5:
            color = "white"
        else:
            color = "black"
        ax.annotate(txt, (y[i], x[i]), size=14,
                    ha='center', va='center', color=color)

    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_XS)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_XS)

    ax.set_xscale('log')

    if show_legend:
        legend = ax.legend(scatter_vect, labels=labels)

        for handle in legend.legendHandles:
            handle._sizes = [80]

    make_invisible = True
    if (make_invisible):
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[1].label1.set_visible(False)

    set_disp(title, xlabel, ylabel)

    plt.show()

    return fig


def plot_timeseries_multi_sub2(timeseries_arrays: List[List[Timeseries]], title, xlabel, ylabel, figsize, ndivx, ndiv, xlabels=None):
    matplotlib.style.use('default')
    id = 0

    if not figsize:
        # fig = plt.figure(id, figsize=(8, 6))
        fig = plt.figure(id)
    else:
        fig = plt.figure(id, figsize=figsize)

    n_sub = len(timeseries_arrays)
    ndata = 0

    for (i, timeseries_array) in enumerate(timeseries_arrays):
        set_plot_font()
        plt.subplot(n_sub * 100 + 11 + i)
        plt.grid()
        for ts in timeseries_array:
            ndata = len(ts.x)
            if xlabels:
                x = xlabels
            else:
                x = ts.x
            y = ts.y

            plt.plot(x, y, label=ts.label, color=ts.color)
            if n_sub == 1 or (i == (n_sub - 1)):
                set_disp(title[i], xlabel, ylabel[i])
            else:
                set_disp(title[i], "", ylabel[i])
            plt.legend(loc="upper left")

    # ax.tick_params(axis = 'both', which = 'major', labelsize = FSIZE_LABEL_XS)
    # ax.tick_params(axis = 'both', which = 'minor', labelsize = FSIZE_LABEL_XS)

    ax = plt.gca()
    if ndiv is None:
        if ndivx is not None:
            ndiv = int(ndata / ndivx)

    if ndiv is not None:
        ax.set_xticks(ax.get_xticks()[::ndiv])

    # yint = []
    # locs, labels = plt.xticks()
    # yint = x
    # plt.xticks(yint)

    fig = plt.gcf()
    plt.show()
    return fig


def set_plot_font(size=FSIZE_LABEL_XS):
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)
    plt.rc('legend', fontsize=size)


def plot_timeseries_multi(timeseries_array: List[Timeseries], title, xlabel, ylabel, separate):
    matplotlib.style.use('default')
    id = 0

    fig = None

    if not separate:
        fig = plt.figure(id)

    set_plot_font()

    for ts in timeseries_array:
        if separate:
            fig = plt.figure(id)
        id += 1
        x = ts.x
        y = ts.y
        plt.plot(x, y, label=ts.label, color=ts.color)

        if separate:
            set_disp(title, xlabel, ylabel)
            plt.legend()
            plt.show(block=False)

    if not separate:
        set_disp(title, xlabel, ylabel)
        plt.legend()
        fig = plt.gcf()
        plt.show()

    if separate:
        plt.show()

    return fig


def stem_timeseries_multi(timeseries_array: List[Timeseries], title, xlabel, ylabel, separate):
    matplotlib.style.use('default')
    fig = plt.figure()

    bottom = None
    for ts in timeseries_array:
        for y in ts.y:
            if bottom is None or y < bottom:
                bottom = y

    for ts in timeseries_array:
        plt.stem(ts.x, ts.y, label=ts.label, bottom=bottom)

    set_disp(title, xlabel, ylabel)

    plt.legend()
    fig = plt.gcf()

    plt.show()

    return fig


def plot_timeseries_ax(timeseries: Timeseries, title, xlabel, ylabel, fig, ax, vlines):
    set_plot_font()

    ax.plot(timeseries.x, timeseries.y)
    # set_disp(title, xlabel, ylabel)

    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, ymin=0, ymax=1, c="coral", ls="-")

    set_disp_ax(ax, title, xlabel, ylabel)

    # plt.legend()
    # plt.show()
    return fig


def plot_timeseries(timeseries: Timeseries, title, xlabel, ylabel):
    fig = plt.figure()
    plt.plot(timeseries.x, timeseries.y)
    set_disp(title, xlabel, ylabel)

    # sio = BytesIO()
    # fig.savefig(sio, format='png')
    # sio.seek(0)
    # return sio

    # plt.show()
    # plt.close()

    plt.legend()

    fig = plt.gcf()

    plt.show()

    return fig


def save_figure(fig, file):
    fig.savefig(file, dpi=300)


def set_disp_ax(ax, title, xlabel, ylabel):
    if title:
        ax.set_title(title,  fontsize=FSIZE_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FSIZE_LABEL)


def set_disp(title, xlabel, ylabel):
    if title:
        plt.gca().set_title(title, fontsize=FSIZE_TITLE, pad=10)
    if xlabel:
        plt.xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        plt.ylabel(ylabel, fontsize=FSIZE_LABEL)


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel('y1')

# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('y2', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')

# plt.savefig('images/two-scales-5.png')

def plot_barchart_multi(bss: List[Barseries], xlabel, ylabel, title, xlabels, limits):
    return plot_barchart_multi_core(bss, xlabel, ylabel, title, xlabels, limits, None, None, True, None, 0, None)[0]
    # 0.155
    # return plot_barchart_multi_core(bss, xlabel, ylabel, title, xlabels, top, None, None, True, -0.125, 0, None)[0]


def plot_barchart_multi_dual(bss1: List[Barseries], bss2: List[Barseries], xlabel, ylabel1, ylabel2, title, xlabels, limits: List[List[int]], show):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    if limits is None:
        limits = [None, None]

    fig, ax1 = plot_barchart_multi_core(
        bss1, xlabel, ylabel1, title, xlabels, limits[0], fig, ax1, False, -0.155, 2, "upper left")

    for b in bss2:
        b.color = "red"

    ax2 = ax1.twinx()
    fig, _ = plot_barchart_multi_core(
        bss2, xlabel, ylabel2, title, xlabels, limits[1], fig, ax2, True, 0.155, 2, "upper right")

    return fig


def plot_barchart_multi_core(bss: List[Barseries], xlabel, ylabel, title, xlabels, limits: List[int], fig, ax, show, offset, bcount, legend_loc):

    # create plot
    if fig is None or ax is None:
        print("creating new figure")
        fig, ax = plt.subplots()

    # ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_XS)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_XS)

    n_groups = len(bss)

    if bcount != 0:
        bar_width = 1 / (bcount + 1)
    else:
        bar_width = 1 / (n_groups+1)

    if offset is None:
        # offset = -1 / (n_groups * 2 * bar_width + 1)
        if n_groups == 2:
            offset = bar_width / 2
        else:
            offset = -bar_width / 2

    # if n_groups == 1:
    #     bar_width = 1

    opacity = OPACITY

    low = None
    high = None

    for i in range(n_groups):

        index = np.arange(len(bss[i].data))

        # print(bss[i].data)

        low1 = min(bss[i].data)
        high1 = max(bss[i].data)

        if low is None:
            low = low1
            high = high1

        if low1 < low:
            low = low1
        if high1 > high:
            high = high1

        rb = plt.bar(
            index + offset + i * bar_width,
            bss[i].data,
            bar_width,
            alpha=opacity,
            color=bss[i].color,
            label=bss[i].label,
            zorder=3)

    plt.xlabel(xlabel, fontsize=FSIZE_LABEL)
    plt.ylabel(ylabel, fontsize=FSIZE_LABEL)
    plt.title(title, fontsize=FSIZE_TITLE)

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)

    if n_groups == 1:
        plt.xticks(index, xlabels)
    else:
        plt.xticks(index + bar_width, xlabels)

    if not legend_loc:
        legend_loc = "upper left"

    plt.legend(loc=legend_loc, fontsize=FSIZE_LABEL_XS)

    ax.grid(zorder=0)

    print("low limit: ", low)
    print("high limit: ", high)
    # plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
    # plt.ylim([math.ceil(low-0.005*(high-low)), math.ceil(high+0.005*(high-low))])
    # plt.ylim([low, high])

    # kscale = 0.25
    kscale = 0.01

    if limits is not None:
        low = limits[0]
        high = limits[1]
    else:
        high += kscale * high
        low -= kscale * low

    plt.ylim([low, high])

    # set_fontsize()
    plt.tight_layout()

    if show:
        print("show")
        plt.show()

    return fig, ax


def plot_barchart_multi_core_raw(data, colors, labels, xlabel, ylabel, title, xlabels, limits, show, offset, bcount, legend_loc):

    fig, ax = plt.subplots(figsize=(10, 8))

    # ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_XS)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_XS)

    n_groups = len(data)

    if bcount != 0:
        bar_width = 1 / (bcount + 1)
    else:
        bar_width = 1 / (n_groups + 1)

    bar_width = 1

    if offset is None:
        # offset = -1 / (n_groups * 2 * bar_width + 1)
        if n_groups == 2:
            offset = bar_width / 2
        else:
            # offset = -bar_width / n_groups
            offset = -1 / ((n_groups + 1) / 2) - bar_width
            # offset = 0
            # offset = -1 / ((n_groups + 1) / 2) + bar_width*3/2
            # pass
    offset = 0

    opacity = OPACITY

    low = None
    high = None

    for i in range(n_groups):

        print("plotting index: " + str(i))

        index = np.arange(len(data[i]))

        if len([d for d in data[i] if d != 0]) != 0:
            low1 = min([d for d in data[i] if d != 0])
        else:
            low1 = None

        high1 = max(data[i])

        if low is None:
            low = low1
            high = high1

        if low1 is not None:
            if low1 < low:
                low = low1
        if high1 is not None:
            if high1 > high:
                high = high1

        rb = plt.bar(
            index + offset,
            data[i],
            bar_width,
            alpha=opacity,
            color=colors[i] if colors is not None else None,
            label=labels[i],
            zorder=3)

    plt.xlabel(xlabel, fontsize=FSIZE_LABEL)
    plt.ylabel(ylabel, fontsize=FSIZE_LABEL)
    plt.title(title, fontsize=FSIZE_TITLE)

    if n_groups == 1:
        plt.xticks(index, xlabels)
    else:
        plt.xticks(index + bar_width, xlabels)

    if not legend_loc:
        legend_loc = "upper left"

    plt.legend(loc=legend_loc, fontsize=FSIZE_LABEL_XS)

    plt.xticks(rotation=0)

    ax.grid(zorder=0)

    print("low limit: ", low)
    print("high limit: ", high)
    # plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
    # plt.ylim([math.ceil(low-0.005*(high-low)), math.ceil(high+0.005*(high-low))])
    # plt.ylim([low, high])

    # kscale = 0.25
    kscale = 0.1

    if limits is not None:
        low = limits[0]
        high = limits[1]
    else:
        high += kscale * high
        low -= kscale * low

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylim([low, high])

    # set_fontsize()
    plt.tight_layout()

    if show:
        print("show")
        plt.show()

    return fig, ax


def set_fontsize():
    # SMALL_SIZE = 20
    # MEDIUM_SIZE = 24
    # BIGGER_SIZE = 28

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)

    matplotlib.rcParams.update({'font.size': 16})


def plot_barchart(labels, values, xlabel, ylabel, title, color, limits, figsize):
    matplotlib.style.use('default')
    id = 0

    if not figsize:
        # fig = plt.figure(id, figsize=(8, 6))
        fig = plt.figure(id)
    else:
        fig = plt.figure(id, figsize=figsize)

    y_pos = np.arange(len(labels))
    plt.rc('axes', axisbelow=True)
    plt.bar(y_pos, values, align='center', alpha=OPACITY, color=color)
    plt.xticks(y_pos, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    low = min(values)
    high = max(values)
    if limits is not None:
        plt.ylim(limits[0], limits[1])
    else:
        plt.ylim([math.ceil(low-0.5*(high-low)),
                  math.ceil(high+0.5*(high-low))])
        # plt.ylim([math.ceil(low-0.005*(high-low)), math.ceil(high+0.005*(high-low))])
        # plt.ylim([low - 0.5*low, high + 0.5*high])
    plt.grid(zorder=0)
    set_disp(title, xlabel, ylabel)
    fig = plt.gcf()
    plt.show()
    return fig


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", scale=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap

    # ax.figure.clim(0, 1)

    im = ax.imshow(data, **kwargs)

    # ax.figure.clim(0, 1)

    if scale is not None:
        for im in plt.gca().get_images():
            im.set_clim(scale[0], scale[1])

    # Create colorbar
    cbar = None
    if cbarlabel is not None:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=False, bottom=True,
    #                labeltop=False, labelbottom=True)

    ax.tick_params(top=False, bottom=False, left=False,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
    #          rotation_mode="anchor")

    # plt.setp(ax.get_yticklabels(), rotation=30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_matrix_cmap(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels, scale=None):

    min_val, max_val = elements[0].val, elements[0].val

    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
    intersection_matrix = np.zeros((xsize, ysize))

    # print(intersection_matrix)
    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val

    fig, ax = plt.subplots()

    # ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_XS)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_XS)

    cmap = "RdYlGn"
    cmap = "viridis"
    cmap = "YlGnBu"

    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap=cmap, cbarlabel="", scale=scale)

    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    def millions(x, pos):
        'The two args are the value and tick position'
        return '%.1f' % (x)

    texts = annotate_heatmap(im, valfmt=millions)

    set_disp(title, xlabel, ylabel)

    fig.tight_layout()
    plt.show()

    return fig


def plot_matrix_cmap_plain(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels, scale=None, figsize=None):

    min_val, max_val = elements[0].val, elements[0].val

    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
    intersection_matrix = np.zeros((xsize, ysize))

    # print(intersection_matrix)

    # intersection_matrix = np.zeros((xsize, ysize + 2))

    for row in range(xsize):
        for col in range(ysize):
            intersection_matrix[row][col] = -1

    for e in elements:
        if e.val == 0:
            e.val = -1

        intersection_matrix[e.i][e.j] = e.val
        # print(e.val)
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val

    # intersection_matrix[0][ysize] = 0
    # intersection_matrix[0][ysize+1] = 1

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    cmap = "RdYlGn"
    cmap = "viridis"
    cmap = "YlGnBu"
    cmap = "Blues"

    # color_map = plt.cm.get_cmap('Blues')
    # cmap = color_map

    cmap = copy.copy(plt.get_cmap(cmap))

    # modify colormap
    alpha = OPACITY
    colors = []
    for ind in range(cmap.N):
        c = []
        # print(cmap(ind))
        # quit()
        c = list(cmap(ind))
        rgb = [c[0], c[1], c[2]]
        hsv = list(rgb2hsv(rgb[0], rgb[1], rgb[2]))
        hsv[0] += 10
        # print(hsv)
        # quit()
        rgb = hsv2rgb(hsv[0], hsv[1], hsv[2])
        c[0] = rgb[0]
        c[1] = rgb[1]
        c[2] = rgb[2]
        c[3] = alpha

        colors.append(tuple(c))

    cmap = matplotlib.colors.ListedColormap(colors, name='my_name')

    cmap.set_under('white', -1)

    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_S)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_S)

    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap=cmap, cbarlabel=None, scale=scale)

    set_disp(title, xlabel, ylabel)

    fig.tight_layout()
    plt.show()

    return fig


def get_n_ax(n, figsize=None, height_ratios=None):
    if figsize:
        # print(n)
        if height_ratios is not None:
            fig, ax = plt.subplots(nrows=n, ncols=1, figsize=figsize, gridspec_kw={
                                   'height_ratios': height_ratios})
        else:
            fig, ax = plt.subplots(nrows=n, ncols=1, figsize=figsize)

        # fig = plt.figure(figsize=figsize)
        # gs = GridSpec(n, 1)
        # ax = []
        # for row in range(n):
        #     ax1 = plt.subplot(gs[row, 0])
        #     ax.append(ax1)
    else:
        fig, ax = plt.subplots(nrows=n, ncols=1)
    return fig, ax


def plot_matrix_cmap_plain_ax(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels, scale, fig, ax, annotate, cmap):
    min_val, max_val = elements[0].val, elements[0].val
    intersection_matrix = np.zeros((xsize, ysize))
    print(np.shape(intersection_matrix))

    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
        # print(e.i, e.j, e.val)
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val

    # cmap = "RdYlGn"
    # cmap = "Blues"

    ax.tick_params(axis='both', which='major', labelsize=FSIZE_LABEL_XS)
    ax.tick_params(axis='both', which='minor', labelsize=FSIZE_LABEL_XS)

    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap=cmap, cbarlabel=None, scale=scale, aspect=1.3)

    # forceAspect(im, ax, 1.0)

    def millions(x, pos):
        'The two args are the value and tick position'
        return '%d' % (x)
        # return str(x)

    if annotate:
        texts = annotate_heatmap(im, valfmt=millions)

    set_disp(title, xlabel, ylabel)


def show_fig(fig):
    fig.tight_layout()
    plt.show()

    return fig


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q

    return r, g, b


def rgb2hsv(r, g, b):
    r, g, b = r, g, b
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def create_discrete_cmap(data):
    # cmap = discrete_cmap(len(data)+2, 'hsv')
    # cmap = discrete_cmap(len(data) + 2, 'nipy_spectral')
    cmap = discrete_cmap(len(data) + 2, 'jet')
    return cmap


def plot_map_features(place_coords, item_coords, colors, starts, ends, zoom_out, labels, sizes, title):
    # Plotting of the routes in matplotlib.
    figsize = (10, 8)

    fig = plt.figure(figsize=figsize)

    set_plot_font()

    ax = fig.add_subplot(111)

    icon_path = "signal.svg"
    doc = minidom.parse(icon_path)  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]

    smiley = parse_path("""m 739.01202,391.98936 c 13,26 13,57 9,85 -6,27 -18,52 -35,68 -21,20 -50,23 -77,18 -15,-4 -28,-12 -39,-23 -18,-17 -30,-40 -36,-67 -4,-20 -4,-41 0,-60 l 6,-21 z m -302,-1 c 2,3 6,20 7,29 5,28 1,57 -11,83 -15,30 -41,52 -72,60 -29,7 -57,0 -82,-15 -26,-17 -45,-49 -50,-82 -2,-12 -2,-33 0,-45 1,-10 5,-26 8,-30 z M 487.15488,66.132209 c 121,21 194,115.000001 212,233.000001 l 0,8 25,1 1,18 -481,0 c -6,-13 -10,-27 -13,-41 -13,-94 38,-146 114,-193.000001 45,-23 93,-29 142,-26 z m -47,18 c -52,6 -98,28.000001 -138,62.000001 -28,25 -46,56 -51,87 -4,20 -1,57 5,70 l 423,1 c 2,-56 -39,-118 -74,-157 -31,-34 -72,-54.000001 -116,-63.000001 -11,-2 -38,-2 -49,0 z m 138,324.000001 c -5,6 -6,40 -2,58 3,16 4,16 10,10 14,-14 38,-14 52,0 15,18 12,41 -6,55 -3,3 -5,5 -5,6 1,4 22,8 34,7 42,-4 57.6,-40 66.2,-77 3,-17 1,-53 -4,-59 l -145.2,0 z m -331,-1 c -4,5 -5,34 -4,50 2,14 6,24 8,24 1,0 3,-2 6,-5 17,-17 47,-13 58,9 7,16 4,31 -8,43 -4,4 -7,8 -7,9 0,0 4,2 8,3 51,17 105,-20 115,-80 3,-15 0,-43 -3,-53 z m 61,-266 c 0,0 46,-40 105,-53.000001 66,-15 114,7 114,7 0,0 -14,76.000001 -93,95.000001 -76,18 -126,-49 -126,-49 z""")
    smiley = parse_path(path_strings[0])
    # for ps in path_strings:
    #     pp = parse_path(ps)
    #     codes = pp.codes
    #     verts = pp.vertices
    #     codes = codes.tolist()
    #     verts = verts.tolist()
    #     # quit()
    # smiley = matplotlib.Path(verts, codes)
    smiley.vertices -= smiley.vertices.mean(axis=0)

    marker = smiley
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))

    valid_series = []
    for i, item in enumerate(item_coords):
        if len(item_coords[item]) > 0:
            valid_series.append(i)

    markerface = ['y.', 'g.', 'b.']
    for i, item in enumerate(item_coords):
        if len(item_coords[item]) > 0:
            clat, clon = zip(*[c for c in item_coords[item]])
            # markerface[i % len(colors)]
            ax.plot(clon, clat, '.', color=colors[i], markersize=sizes[i])

    if len(place_coords) > 0:
        # Plot all the nodes as black dots.
        clat, clon = zip(*[c for c in place_coords])
        ax.plot(clon, clat, 'm.', color=colors[len(
            colors)-1], marker=marker, markersize=20)

    # ax.plot(clon, clat, 'b.', markersize=20)

    # ax.plot(clon, clat, 'b.', markersize=20)
    # plot the routes as arrows

    labels_valid = []

    for idx in valid_series:
        labels_valid.append(labels[idx])

    # labels_valid.append("O")

    lgnd = ax.legend(labels_valid)

    # change the marker size manually
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(10)
        handle._legmarker.set_markersize(10)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # example of how to zoomout by a factor of 0.1
    factor = zoom_out
    new_xlim = (xlim[0] + xlim[1])/2 + np.array((-0.5, 0.5)) * \
        (xlim[1] - xlim[0]) * (1 + factor)
    ax.set_xlim(new_xlim)
    new_ylim = (ylim[0] + ylim[1])/2 + np.array((-0.5, 0.5)) * \
        (ylim[1] - ylim[0]) * (1 + factor)
    ax.set_ylim(new_ylim)

    plt.grid(zorder=0)
    set_disp(title, "longitude", "latitude")

    plt.show()
    return fig
