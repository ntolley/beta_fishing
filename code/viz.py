import matplotlib.pyplot as plt
import numpy as np

def plot_spikes_raster(spike_times, spike_gids, spike_types, gid_ranges, trial_idx=None, ax=None, show=True):
    """Plot the aggregate spiking activity according to cell type.

    Parameters
    ----------
    cell_response : instance of CellResponse
        The CellResponse object from net.cell_response
    trial_idx : int | list of int | None
        Index of trials to be plotted. If None, all trials plotted
    ax : instance of matplotlib axis | None
        An axis object from matplotlib. If None, a new figure is created.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : instance of matplotlib Figure
        The matplotlib figure object.
    """

    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']
    cell_type_colors = {'L5_pyramidal': 'r', 'L5_basket': 'b',
                        'L2_pyramidal': 'g', 'L2_basket': 'w'}
    cell_type_labels = {'L5_pyramidal': 'L5e', 'L5_basket': 'L5i',
                         'L2_pyramidal': 'L2e', 'L2_basket': 'L2i'}

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    ypos = 0
    events = []
    for cell_type in cell_types:
        cell_type_gids = np.unique(spike_gids[spike_types == cell_type])
        cell_type_times, cell_type_ypos = [], []
        for gid in cell_type_gids:
            gid_time = spike_times[spike_gids == gid]
            cell_type_times.append(gid_time)
            cell_type_ypos.append(ypos)
            ypos = ypos - 1

        if cell_type_times:
            events.append(
                ax.eventplot(cell_type_times, lineoffsets=cell_type_ypos,
                             color=cell_type_colors[cell_type],
                             label=cell_type_labels[cell_type], linelengths=5))

    ax.legend(handles=[e[0] for e in events], loc=1)
    ax.set_facecolor('k')
    ax.set_xlabel('Time (ms)')
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(left=0)

    return ax.get_figure()