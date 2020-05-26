import matplotlib.pyplot as plt


def plot_crosshair(x: float, y: float, ax=None, color='r'):
    """
    Plot Crosshair and deduce limits automatically from axis

    Args:
        x: x coordinate
        y: y coordinate
        ax: axis where to put crosshair
        color: colour as specified by matplotlib

    Returns:
        None
    """

    if ax is None:
        ax = plt.gca()

    """Get ax limits"""
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    ax.hlines(y, *xl, colors=color)
    ax.vlines(x, *yl, colors=color)
