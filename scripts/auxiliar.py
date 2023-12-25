import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.colors as colors
from datetime import datetime
from scipy.stats import chi2
from scipy.optimize import fsolve

# logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

sigma = "$\sigma$"

def find_files_in_dir(directory):
    """
    Find all files in a directroy and all subdirectories inside
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        # iterate over all files inside one directory
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list



def createdir(path):
    """
    Create a directory if doesn't exists
    """
    # checking if exists first
    if not os.path.exists(path):
        os.makedirs(os.path.join(path), exist_ok=True)    
        
        
        
def find_nearest_value(array, value):
    """
    Finding the nearest integer inside a array
    """
    idx = (np.abs(np.array(array) - value)).argmin()
    return array[idx]        



def transparent_cmap(cmap, ranges=[0,1]):
    '''
    Retuns a colormap object tuned to transparent
    '''
    
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(*ranges, ncolors)
    
    # building the colormap
    return colors.LinearSegmentedColormap.from_list(name='cmap', colors=color_array)



def move_files(source_folder, destination_folder):
    '''
    Function to move files from one directory to another
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
    A function that try to delete a file
    '''
    
    try:
        os.rmdir(directory_path)
        logger.debug(f"Directory '{directory_path}' deleted successfully.")
        
    except OSError as error:
        logger.debug(f"Error deleting directory '{directory_path}': {error}")


        
def params(n=15):
    '''
    Function to set standard parameters for matplotlib
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
    '''    
    return colors.LinearSegmentedColormap.from_list('',  cols)


def plot_colorbar(fig, ax, array, cmap, label=""):
    
    norm = mpl.colors.Normalize(vmin=min(array), vmax=max(array))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label)    


c1 = (5/255,5/255,153/255)
c2 = (102/255,0/255,204/255)
c3 = (255/255,51/255,204/255)
c4 = (204/255,0/255,0/255)
c5 = (255/255,225/255,0/255)

predC = [c1, c2, c3, c4, c5]

def color_cr(x, col=predC):
    
    '''
    Function to create a color gradient of 5 colors in this case
    
    Input
    ------------
    --n: float
            the value from 0 to 1 to assign a colour
            
    Output
    ------------
    --r, g, b: float
            the rgb values for the color to assign
    
    '''
    size = len(col)
    size_bins = size -1 
    
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
        print(f'Input {x} should be a float in range [0 , 1]')
        
    if x > 1 or x < 0:
        print(f'Input {x} should be in range [0 , 1]')
    
    for i in range(size_bins):
        if x >= i/size_bins and x <= (i+1)/size_bins:
            xeff = x - i/size_bins
            r = COLORS[i][0] * (1 - size_bins * xeff) + COLORS[i+1][0] * size_bins * xeff
            g = COLORS[i][1] * (1 - size_bins * xeff) + COLORS[i+1][1] * size_bins * xeff
            b = COLORS[i][2] * (1 - size_bins * xeff) + COLORS[i+1][2] * size_bins * xeff
            
    return (r, g, b)



def get_colors_multiplot(array, COLORS=predC, ran=None):
    
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
    norm   = mpl.colors.Normalize(vmin=np.min(array), vmax=np.max(array))
    colors = mpl.cm.ScalarMappable(norm, cmap).to_rgba(array)    
    
    return norm, colors

def calculate_chi2_pvalue_const(y, uy, sys_error=0):
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = (y/uncertainty**2).sum() / (1/uncertainty**2).sum()
    mean_y_err = np.sqrt(1/np.sum(1/uncertainty**2))
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def calculate_chi2_pvalue_fun(x, y, uy, f, params, sys_error=0):
    x, y, uy = np.array(x), np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    
    mean_y     = f(params, x)
    
    chi2_value = np.sum((y - mean_y)**2/uncertainty**2)
    ndf = len(y) - 1
    pvalue = chi2.sf(x=chi2_value, df=ndf)
    return chi2_value, ndf, pvalue

def weighted_average(y, uy, sys_error=0):
    y, uy = np.array(y), np.array(uy)
    
    uncertainty = np.sqrt((sys_error * y)**2 + uy**2)
    return (y/uncertainty**2).sum() / (1/uncertainty**2).sum(), np.sqrt(1/np.sum(1/uncertainty**2))


def sortbased(X, REF):
    return np.sort(REF), np.array([x for ref, x in sorted(zip(REF, X))])