from venv import logger
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.colors as colors
from scipy.stats import chi2

# Some symbols to add easily to text
sigma = "$\sigma$"
mu = "$\mu$"


def find_files_in_dir(directory):
    """
    Find all files in a directory and its subdirectories.

    Args:
        directory (str): The directory to search.

    Returns:
        list: A list of file paths.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]


def createdir(path):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): The path of the directory to be created.

    Returns:
        None
    """
    # checking if exists first
    if not os.path.exists(path):
        os.makedirs(os.path.join(path), exist_ok=True)
                
        
def find_nearest_value(array, value):
    """
    Finding the nearest integer inside an array.

    Parameters:
    array (list or numpy.ndarray): The input array.
    value (int or float): The value to find the nearest integer to.

    Returns:
    int or float: The nearest integer to the given value in the array.
    """
    idx = (np.abs(np.array(array) - value)).argmin()
    return array[idx]


def transparent_cmap(cmap, ranges=[0,1]):
    '''
    Returns a colormap object tuned to transparent.

    Parameters:
        cmap (str): The name of the base colormap.
        ranges (list, optional): The range of transparency values. Defaults to [0, 1].

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The transparent colormap object.
    '''
    
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(*ranges, ncolors)
    
    # building the colormap
    return colors.LinearSegmentedColormap.from_list(name='cmap', colors=color_array)



def move_files(source_folder, destination_folder):
    '''
    Function to move files from one directory to another
    
    Parameters:
        source_folder (str): The path of the source folder containing the files to be moved.
        destination_folder (str): The path of the destination folder where the files will be moved to.
    '''
    
    # iterating over all files inside the folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        # checking if is a file or a folder
        if os.path.isfile(source_path):
            destination_path = os.path.join(destination_folder, filename)
            
            # moving file by file
            shutil.move(source_path, destination_path)


            
def delete_directory(directory_path):
    '''
    Deletes a directory if it exists.

    Parameters:
        directory_path (str): The path of the directory to be deleted.

    Returns:
        None

    Raises:
        OSError: If an error occurs while deleting the directory.
    '''
    
    try:
        os.rmdir(directory_path)
        logger.debug(f"Directory '{directory_path}' deleted successfully.")
        
    except OSError as error:
        logger.debug(f"Error deleting directory '{directory_path}': {error}")


        
def params(n=15):
    '''
    Function to set standard parameters for matplotlib.

    Parameters:
        n (int): Font size for matplotlib.

    '''
    plt.rcParams['font.size'] = n
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['axes.linewidth'] = 1.9
    plt.rcParams['figure.figsize'] = (13, 7)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.8
    plt.rcParams['ytick.major.width'] = 1.8   
    plt.rcParams['lines.markeredgewidth'] = 2
    pd.set_option('display.max_columns', None)


def create_cmap(cols):
    '''
    Create a colormap given an array of colors
    
    Parameters:
        cols (list): List of colors to create the colormap from
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap: The created colormap
    '''    
    return colors.LinearSegmentedColormap.from_list('',  cols)


def plot_colorbar(fig, ax, array, cmap, label=""):
    """
    Add a colorbar to a matplotlib figure.

    Parameters:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object where the colorbar will be added.
    - array: The array of values used to determine the color of each element in the colorbar.
    - cmap: The colormap used to map the values in the array to colors.
    - label: The label for the colorbar.

    Returns:
    None
    """
    norm = mpl.colors.Normalize(vmin=min(array), vmax=max(array))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label)

# Example colors
c1 = (5/255,5/255,153/255)
c2 = (102/255,0/255,204/255)
c3 = (255/255,51/255,204/255)
c4 = (204/255,0/255,0/255)
c5 = (255/255,225/255,0/255)
predC = [c1, c2, c3, c4, c5] # Defautl colors for the plots

def color_cr(x, col=predC):
    '''
    Function to create a color gradient of 5 colors in this case
    
    Parameters:
        x (float): The value from 0 to 1 to assign a color
        col (list): List of tuples or list of strings, optional. The list of colors to use for the gradient. Each color can be specified as a tuple of RGB values or as a string representing a named color. Default is predC.
            
    Returns:
        tuple: The RGB values for the color to assign
    
    Raises:
        ValueError: If the input x is not a float in the range [0, 1]
    '''
    size = len(col)
    size_bins = size - 1 
    
    COLORS = []
    for i in range(size):
        
        if type(col[i]) == str:
            c = colors.to_rgba(col[i])
        else:
            c = col[i]
        
        COLORS.append(c)
    
    try:
        x = float(x)
    except ValueError:
        raise ValueError(f'Input {x} should be a float in range [0 , 1]')
        
    if x > 1 or x < 0:
        raise ValueError(f'Input {x} should be in range [0 , 1]')
    
    for i in range(size_bins):
        if x >= i/size_bins and x <= (i+1)/size_bins:
            xeff = x - i/size_bins
            r = COLORS[i][0] * (1 - size_bins * xeff) + COLORS[i+1][0] * size_bins * xeff
            g = COLORS[i][1] * (1 - size_bins * xeff) + COLORS[i+1][1] * size_bins * xeff
            b = COLORS[i][2] * (1 - size_bins * xeff) + COLORS[i+1][2] * size_bins * xeff
            
    return (r, g, b)



def get_colors_multiplot(array, COLORS=predC, ran=None):
    """
    Returns a list of colors corresponding to each element in the input array.
    
    Parameters:
    - array: list or array-like object containing the values for which colors are needed.
    - COLORS: list of colors to choose from. Default is predC.
    - ran: tuple (min, max) specifying the range of values. Default is None, in which case the minimum and maximum values of the array are used.
    
    Returns:
    - colors: list of colors corresponding to each element in the input array.
    """
    
    # getting the color of each run
    colors = []
   
    if ran != None:
        m = ran[0]
        M = ran[1]
    else:
        m = min(array)
        M = max(array)
    
    for i in range(len(array)):
        
        if array[i] > M:
            colors.append(color_cr(1, COLORS))
        elif array [i] < m:
            colors.append(color_cr(0, COLORS))
        else:
            normalized_value = (array[i] - m) / (M - m)
            colors.append(color_cr(normalized_value, COLORS))   
    
    return colors

def get_cmap_colors(array, cmap):
    """
    Get normalized values and colors from an array using a specified colormap.

    Parameters:
    array (numpy.ndarray): The input array.
    cmap (matplotlib.colors.Colormap): The colormap to use.

    Returns:
    tuple: A tuple containing the normalization object and the array of colors.
    """

    norm   = mpl.colors.Normalize(vmin=np.min(array), vmax=np.max(array))
    colors = mpl.cm.ScalarMappable(norm, cmap).to_rgba(array)    
    
    return norm, colors

def calculate_chi2_pvalue_const(y, uy, sys_error=0):
    """
    Calculate the chi-square value, degrees of freedom, and p-value for a constant fit.

    Parameters:
    y (array-like): Observed values.
    uy (array-like): Uncertainties of the observed values.
    sys_error (float, optional): Systematic error. Default is 0.

    Returns:
    tuple: A tuple containing the chi-square value, degrees of freedom, and p-value.
    """
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = (y/uncertainty**2).sum() / (1/uncertainty**2).sum()
    mean_y_err = np.sqrt(1/np.sum(1/uncertainty**2))
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def calculate_chi2_pvalue_fun(x, y, uy, f, params, sys_error=0):
    """
    Calculate the chi-square value, degrees of freedom, and p-value for a given set of data and model.

    Parameters:
    - x: array-like, x-values of the data
    - y: array-like, y-values of the data
    - uy: array-like, uncertainties of the y-values
    - f: function, model function that takes parameters and x-values as inputs
    - params: array-like, parameters for the model function
    - sys_error: float, systematic error (default: 0)

    Returns:
    - chi2_value: float, chi-square value
    - ndf: int, degrees of freedom
    - pvalue: float, p-value
    """
    x, y, uy = np.array(x), np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = f(params, x)
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def weighted_average(y, uy, sys_error=0):
    """
    Calculate the weighted average of a set of values.

    Parameters:
    y (array-like): The values to be averaged.
    uy (array-like): The uncertainties associated with each value.
    sys_error (float, optional): The systematic error to be considered. Default is 0.

    Returns:
    float: The weighted average.
    float: The uncertainty of the weighted average.
    """
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    return (y/uncertainty**2).sum() / (1/uncertainty**2).sum(), np.sqrt(1/np.sum(1/uncertainty**2))


def sortbased(X, REF):
    """
    Sorts the array REF in ascending order and rearranges the array X based on the sorted REF values.

    Parameters:
    X (array-like): The array to be rearranged based on REF.
    REF (array-like): The reference array used for sorting.

    Returns:
    tuple: A tuple containing two arrays. The first array is the sorted REF, and the second array is X rearranged based on the sorted REF values.
    """
    return np.sort(REF), np.array([x for ref, x in sorted(zip(REF, X))])