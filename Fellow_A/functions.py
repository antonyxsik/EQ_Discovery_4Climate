import os
import re

import xarray as xr
import netCDF4 as nc
import h5netcdf

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

import pysr
from pysr import PySRRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def list_directories_files(path):
    """
    provides all files and directories from the LES simulations (they start with Ug)

    Parameters:
    - path (str): The path to the LES simulation data.

    Returns:
    - directories (list): A list of all directories starting with 'Ug'.
    - files (list): A list of all files starting with 'Ug'.
    """

    # all items in the given path
    items = os.listdir(path)
    
    # find the directories starting with 'Ug'
    directories = [item for item in items if os.path.isdir(os.path.join(path, item)) and item.startswith('Ug')]
    # find the files starting with 'Ug'
    files = [item for item in items if os.path.isfile(os.path.join(path, item)) and item.startswith('Ug')]
    
    return directories, files


def time_average(data, timeavg):
    """
    Averages an array over specified number of time steps. Works for both 1D and higher-dimensional arrays.

    Parameters:
    - data (numpy.ndarray): The input data array. Expected shapes are either (ntime,) or (ntime, nz).
    - timeavg (int): The number of time steps over which to average.

    Returns:
    - numpy.ndarray: The averaged data array.
    """
    ntime = data.shape[0]
    nchunks = ntime // timeavg
    truncated_data = data[:nchunks * timeavg]

    if data.ndim == 1:
        # for 1D arrays, reshape to (nchunks, timeavg)
        reshaped_data = truncated_data.reshape(nchunks, timeavg)
    else:
        # for 2D arrays, old method with reshaping
        nz = data.shape[1]
        reshaped_data = truncated_data.reshape(nchunks, timeavg, nz)

    # compute the mean along the new time axis 
    averaged_data = reshaped_data.mean(axis=1)

    return averaged_data


def make_variables(path, items, time_avg):
    """
    Extracts the variables of interest from the LES simulations and averages them over specified number of time steps.

    Parameters:
    - path (str): The path to the LES simulation data.
    - items (list): A list of files containing the LES simulation data.
    - time_avg (int): The number of time steps over which to average.

    Returns:
    - all variables (numpy.ndarray)
    """

    # Initialize empty lists to accumulate results
    sigma_th = []
    sigma_2 = []
    Theta = []
    wtheta = []
    wwtheta = []
    rdstr = []
    transport = []

    for item in items:
        ds_stat = nc.Dataset(os.path.join(path, item), mode='r')
        
        sigma_th_temp = time_average(ds_stat.groups['thermo']['th_2'][:], time_avg)  # covariance of theta
        sigma_2_temp = time_average(ds_stat.groups['default']['w_2'][:], time_avg)   # covariance of w
        Theta_temp = time_average(ds_stat.groups['thermo']['th'][:], time_avg)       # domain mean theta
        wtheta_temp = time_average(ds_stat.groups['thermo']['th_flux'][:], time_avg) # heat flux
        wwtheta_temp = time_average(ds_stat.groups['budget']['wwtheta'][:], time_avg) # third moment, covariance between wtheta and w
        rdstr_temp = time_average(ds_stat.groups['budget']['bw_rdstr'][:], time_avg) # pressure redistribution term
        transport_temp = time_average(ds_stat.groups['budget']['bw_pres'][:], time_avg) # pressure transport term

        # Append the results to the respective lists
        sigma_th.append(sigma_th_temp)
        sigma_2.append(sigma_2_temp)
        Theta.append(Theta_temp)
        wtheta.append(wtheta_temp)
        wwtheta.append(wwtheta_temp)
        rdstr.append(rdstr_temp)
        transport.append(transport_temp)

    # Concatenate the results along the time axis
    sigma_th = np.concatenate(sigma_th, axis=0)
    sigma_2 = np.concatenate(sigma_2, axis=0)
    Theta = np.concatenate(Theta, axis=0)
    wtheta = np.concatenate(wtheta, axis=0)
    wwtheta = np.concatenate(wwtheta, axis=0)
    rdstr = np.concatenate(rdstr, axis=0)
    transport = np.concatenate(transport, axis=0)

    return sigma_th, sigma_2, Theta, wtheta, wwtheta, rdstr, transport


def reshape_variables(variable):
    reshaped = (variable[:, :-1] + variable[:, 1:]) / 2.0
    return reshaped


def extract_ug_q(filename):
    # Extracting 'ug' and 'q' from the filename
    match = re.search(r'Ug(\d+)Q(\d+)', filename)
    ug = int(match.group(1))
    q = int(match.group(2))

    return ug, q

def make_constants(path, items, time_avg):
    """
    Extracts the constants of interest from the LES simulations and averages them over specified number of time steps.

    Parameters:
    - path (str): The path to the LES simulation data.
    - items (list): A list of files containing the LES simulation data.
    - time_avg (int): The number of time steps over which to average.

    Returns:
    - all constants (numpy.ndarray)
    """
    wtheta_surface = []
    pbl_height = []
    wstar = []
    theta_star = []
    scaling = []
    ustar = []
    grr = 9.8
    T_0 = 300
    beta = grr/T_0
    ug = []
    q = []

    for item in items:
        ds_stat = nc.Dataset(os.path.join(path, item), mode='r')

        wtheta_surface_raw = ds_stat.groups['thermo']['th_flux'][:,0] 
        wtheta_surface_temp = time_average(wtheta_surface_raw, time_avg)

        pbl_height_raw = ds_stat.groups['thermo'].variables['zi'][:] 
        pbl_height_temp = time_average(pbl_height_raw, time_avg)

        wstar_raw = np.power( beta * (wtheta_surface_raw) * pbl_height_raw , 1/3) 
        wstar_temp = time_average(wstar_raw, time_avg)

        theta_star_raw = wtheta_surface_raw / wstar_raw
        theta_star_temp = time_average(theta_star_raw, time_avg)

        scaling_raw = wstar_raw**2 * theta_star_raw / pbl_height_raw
        scaling_temp = time_average(scaling_raw, time_avg)

        ustar_temp = time_average(ds_stat.groups['default'].variables['ustar'][:], time_avg)

        ug_val, q_val = extract_ug_q(item)
        # ug = np.ma.masked_array(np.full(wtheta_surface_temp.shape, ug_val), mask=wtheta_surface_temp.mask)
        # q = np.ma.masked_array(np.full(wtheta_surface_temp.shape, q_val), mask=wtheta_surface_temp.mask)
        ug_repeat = np.full(wtheta_surface_temp.shape, ug_val)
        q_repeat = np.full(wtheta_surface_temp.shape, q_val)

        wtheta_surface.append(wtheta_surface_temp)
        pbl_height.append(pbl_height_temp)
        wstar.append(wstar_temp)
        theta_star.append(theta_star_temp)
        scaling.append(scaling_temp)
        ustar.append(ustar_temp)
        ug.append(ug_repeat)
        q.append(q_repeat)

    wtheta_surface = np.concatenate(wtheta_surface, axis=0)
    pbl_height = np.concatenate(pbl_height, axis=0)
    wstar = np.concatenate(wstar, axis=0)
    theta_star = np.concatenate(theta_star, axis=0)
    scaling = np.concatenate(scaling, axis=0)
    ustar = np.concatenate(ustar, axis=0)
    ug = np.concatenate(ug, axis=0)
    q = np.concatenate(q, axis=0)

    wtheta_surface = wtheta_surface[:,np.newaxis]
    pbl_height = pbl_height[:,np.newaxis]
    wstar = wstar[:,np.newaxis]
    theta_star = theta_star[:,np.newaxis]
    scaling = scaling[:,np.newaxis]
    ustar = ustar[:,np.newaxis]
    ug = ug[:,np.newaxis]
    q = q[:,np.newaxis]

    return wtheta_surface, pbl_height, wstar, theta_star, scaling, ustar, grr, T_0, beta, ug, q


def discover_eqs(path, selected_files, time_avg = 15, indices = np.s_[:, 0:200], difficulty = "medium", normChoice = "None"):

    """
    Performs equation discovery on all selected files provided 
    Goes through the whole workflow including data manipulation, model setup, and fitting

    Parameters:
    - path (str): The path to the LES simulation data.
    - selected_files (list): A list of files containing the LES simulation data.
    - time_avg (int): The number of time steps over which to average.\
    - indices (np.s_[]): The [timestep, height] indices to slice all variables at
    - difficulty (string): Decides the variables to be included in the equation discovery  

    Returns:
    - df_EQ (pandas dataframe): "Hall of fame" dataframe of progressively better fitting equations
    from the PySR software 
    """

    #making variables
    sigma_th, sigma_2, Theta, wtheta, wwtheta, rdstr, transport = make_variables(path, selected_files, time_avg)

    #getting dimensions
    dim_df = nc.Dataset(os.path.join(path, selected_files[0]), mode='r')
    z = dim_df.variables['z'][:]
    zh = dim_df.variables['zh'][:]
    t = dim_df.variables['time'][:]

    # necessary gradients
    dTheta_dz = np.gradient(Theta, z, axis = 1)
    dwwtheta_dz = np.gradient(wwtheta, zh, axis = 1)

    #reshaping to z dim of 384 for all of them
    sigma_2 = reshape_variables(sigma_2)
    wtheta = reshape_variables(wtheta)
    wwtheta = reshape_variables(wwtheta)
    dwwtheta_dz = reshape_variables(dwwtheta_dz)
    rdstr = reshape_variables(rdstr)
    transport = reshape_variables(transport)

    #making constants
    wtheta_surface, pbl_height, wstar, theta_star, scaling, ustar, grr, T_0, beta, ug, q = make_constants(path, selected_files, time_avg)
    tau = pbl_height/wstar

    #computing P value as a residual
    M = (- sigma_2 * dTheta_dz)
    T = - dwwtheta_dz
    B = (beta * sigma_th)
    P = - M - T - B

    # our target is y (residual (P)
    y = P[indices].ravel()

    # all possible variable inputs
    x0 = wtheta[indices].ravel()
    x1 = sigma_th[indices].ravel()
    x2 = sigma_2[indices].ravel()
    x3 = dTheta_dz[indices].ravel()
    x4 = (sigma_2[indices] * dTheta_dz[indices]).ravel()
    x5 = (np.repeat(ug, 384, axis=1))[indices].ravel()
    x6 = (np.repeat(q, 384, axis=1))[indices].ravel()
    x7 = (np.repeat(tau, 384, axis=1))[indices].ravel()
    x8 = (np.repeat(ustar, 384, axis=1))[indices].ravel()

    # output dataframe
    df_y = pd.DataFrame(y, columns=['P'])

    # input dataframe options
    if difficulty == "easy":
        df_X = pd.DataFrame(np.column_stack([x0, x1, x4]), 
                            columns=['wtheta', 'sigma_th', 'Mult'])
    elif difficulty == "medium":
        df_X = pd.DataFrame(np.column_stack([x0, x1, x2, x3]), 
                            columns = ['wtheta', 'sigma_th', 'sigma_2', 'dTheta_dz'])
    elif difficulty == "mediumhard":
        df_X = pd.DataFrame(np.column_stack([x0, x1, x4, x5, x6, x7, x8]), 
                            columns = ['wtheta', 'sigma_th', 'Mult', 'ug', 'q', 'tau', 'ustar'])
    elif difficulty == "hard":
        df_X = pd.DataFrame(np.column_stack([x0, x1, x2, x3, x5, x6, x7, x8]), 
                            columns = ['wtheta', 'sigma_th', 'sigma_2', 'dTheta_dz', 'ug', 'q', 'tau', 'ustar'])
    else:
        print("Please set difficulty to be one of the following: easy, medium, mediumhard, hard")

    # Normalization options
    if normChoice == "none": 
        print("No normalization applied")
    elif normChoice == "minmax": 
        df_X = (df_X - df_X.min()) / (df_X.max() - df_X.min())
        df_y = (df_y - df_y.min()) / (df_y.max() - df_y.min())
        print("Don't forget to unnormalize the coef (minmax)")
    elif normChoice == "zscore":
        df_X = (df_X - df_X.mean()) / df_X.std()
        df_y = (df_y - df_y.mean()) / df_y.std()
        print("Don't forget to unnormalize the coef (zscore)")
    else:
        print("Please set normChoice to be one of the following: none, minmax, zscore")

    model = PySRRegressor(
    niterations = 8000,  # increase me for better results
    maxsize = 28, # allowing for appropriate complexity (x + y has size 3)
    maxdepth = 4, # avoiding deep nesting
    progress = False, # makes the printout less hectic in Jupyter
    binary_operators=["+", "*", "-", "/", "^"],
    # unary_operators=[
    #     "cos",
    #     "exp",
    #     "sin",
    #     "inv(x) = 1/x",
    #     "square",
    #     "cube",
    #     # ^ Custom operator (julia syntax)
    # ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    batching = True,
    batch_size = 250,
    complexity_of_operators = {"*": 1, "+": 1, "-": 1, "^":3,
                             "exp": 3, "sin": 3, "cos": 3, 
                             "inv": 3, "square": 3, "cube": 3},
    constraints = {'^': (-1, 1)},
    # complexity_of_constants = 3,
    # ^ Custom complexity of particular operators and constants
    )

    model.fit(df_X, df_y)

    df_EQ = model.equations_
    
    return df_EQ



def LES_linear_regressor(path, selected_files, time_avg = 15, indices = np.s_[:, 0:200], verbose = True):

    """
    Performs linear regression on all selected files provided 
    Goes through the whole workflow including data manipulation, model setup, fitting, and evaluation

    Parameters:
    - path (str): The path to the LES simulation data.
    - selected_files (list): A list of files containing the LES simulation data.
    - time_avg (int): The number of time steps over which to average.
    - indices (np.s_[]): The [timestep, height] indices to slice all variables at

    Returns:
    - model (sklearn.linear_model): The trained linear regression model
    - X_train, y_train (pd dataframes): Training data 
    - X_test, y_test (pd dataframes): Testing data
    """

    #making variables
    sigma_th, sigma_2, Theta, wtheta, wwtheta, rdstr, transport = make_variables(path, selected_files, time_avg)

    #getting dimensions
    dim_df = nc.Dataset(os.path.join(path, selected_files[0]), mode='r')
    z = dim_df.variables['z'][:]
    zh = dim_df.variables['zh'][:]
    t = dim_df.variables['time'][:]

    # necessary gradients
    dTheta_dz = np.gradient(Theta, z, axis = 1)
    dwwtheta_dz = np.gradient(wwtheta, zh, axis = 1)

    #reshaping to z dim of 384 for all of them
    sigma_2 = reshape_variables(sigma_2)
    wtheta = reshape_variables(wtheta)
    wwtheta = reshape_variables(wwtheta)
    dwwtheta_dz = reshape_variables(dwwtheta_dz)
    rdstr = reshape_variables(rdstr)
    transport = reshape_variables(transport)

    #making constants
    wtheta_surface, pbl_height, wstar, theta_star, scaling, ustar, grr, T_0, beta, ug, q = make_constants(path, selected_files, time_avg)
    tau = pbl_height/wstar

    #computing P value as a residual
    M = (- sigma_2 * dTheta_dz)
    T = - dwwtheta_dz
    B = (beta * sigma_th)
    P = - M - T - B

    # our target is y (residual (P)
    y = P[indices].ravel()
    df_y = pd.DataFrame(y, columns=['P'])

    # all possible variable inputs
    x0 = wtheta[indices].ravel()
    x1 = sigma_th[indices].ravel()
    x4 = (sigma_2[indices] * dTheta_dz[indices]).ravel()
    df_X_mult = pd.DataFrame(np.column_stack([x0, x1, x4]), 
                            columns=['wtheta', 'sigma_th', 'Mult'])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_X_mult, df_y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    coefficients = model.coef_[0]  # Since coef_ returns a 2D array for multi-output, we take the first element
    intercept = model.intercept_[0]  # Similarly, intercept_ returns a 1D array for multi-output


    if verbose:

        print(f"RMSE: {rmse:.7f}")
        print(f"R-squared: {r2:.7f}")

        for col, coef in zip(df_X_mult.columns, coefficients):
            print(f"Coefficient for {col}: {coef:.7f}")
        print(f"Intercept: {intercept:.7f}")

    return model, X_train, X_test, y_train, y_test, rmse, r2, coefficients

def discover_coef_eqs(predictors, coefficient):

    model = PySRRegressor(
    niterations = 500,  # increase me for better results
    maxsize = 10, # allowing for appropriate complexity (x + y has size 3)
    maxdepth = 3, # avoiding deep nesting
    progress = False, # makes the printout less hectic in Jupyter
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        "square",
        "cube",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    complexity_of_operators = {"*": 1, "+": 1, "-": 1,
                             "exp": 3, "sin": 3, "cos": 3, 
                             "inv": 3, "square": 3, "cube": 3},
    # complexity_of_constants = 3,
    # ^ Custom complexity of particular operators and constants
    )

    model.fit(predictors, coefficient)

    df_EQ = model.equations_
    
    return df_EQ

