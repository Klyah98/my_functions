#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


def gini (y_test, proba):
    
    """ y_test - array of true answers, proba - array of corresponding probabilities
    
        returns: data (table to use in gini_plot), g (gini coef) """
    
    data = pd.DataFrame(data=[np.array(y_test), proba]).T
    data = data.sort_values(by=1, axis=0, ascending=False)
    two = [(i+1)/data.shape[0] for i in range(data.shape[0])]
    three = []
    k = 0
    step = 1/sum(data[0])
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == 1:
            k += step
            three.append(k)
        else:
            three.append(k)
    data[2] = two
    data[3] = three
    ideal = (len(data[0]) - sum(data[0]))/(2*len(data[0]))
    S = 0
    x = [0] + list(data[2])
    y = [0] + list(data[3])
    for i in range(data.shape[0] - 1):
        S += (y[i] + y[i+1])/2*(x[i+1] - x[i])
    S = S - 0.5
    g = S/ideal
    return (data, g)


def  gini_plot(data):
    point = sum(data[0])/len(data[0])
    x = [0] + list(data[2])
    y = [0] + list(data[3])

    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.plot([0, 1], [0,1], color='black')
    plt.plot([0, point, 1], [0, 1, 1], color='green')
    plt.show()

    
def bin_bad_rate (dataset, name, default_name):
    
    """ dataset - table, name - name of feature (column) to count bad rate (contains of only ones and zeros),
        default_name - name of column with default flag (contains of only ones and zeros)
        
        returns: bad rate of correspinding column """
    
    nominator = sum(dataset[dataset[name] != 0][default_name])
    denominator = len(dataset[dataset[name] != 0][default_name])
    if denominator == 0:
        return np.nan
    return nominator/denominator


def woe (dataset, name, default_name, IV=False):
    
    """ dataset - table, name - name of feature (column) to count woe (contains of ones and zeros only),
        default_name - name of column with default flag (contains of only ones and zeros)
        IV - whether to return information value term
        
        returns: woe of correspinding column """
    
    default = dataset[dataset[name] == 1][default_name]
    d1 = sum(default)/sum(dataset[default_name])
    d2 = (len(default) - sum(default))/(len(dataset[default_name]) - sum(dataset[default_name]))
    if d1 == 0 or d2 == 0:
        return np.nan
    if IV:
        return np.log(d2/d1), (d2 - d1)*np.log(d2/d1)
    return np.log(d2/d1)


def unify(dataset, names):
    
    """ dataset - table, names - list of names of columns to unify (each contains of ones and zeros only) """
    
    name1 = names[0]
    final_name = names[0]
    list_names = list(dataset.columns)
    j = list_names.index(name1)
    del(names[0])
    for name in tqdm(names):
        step = name.split(':')[-1]
        final_name = final_name + '//' + step
        dataset[name1] = dataset[name1] | dataset[name]
    dataset.rename(columns={name1 : final_name}, inplace=True)
    for i in names:
        dataset.drop(i, axis=1, inplace=True)

        
def cramer (conf):
    
    """ conf - pd.crosstab matrix of two variables
    
        returns: Cramer's V coef """
    
    if conf.shape == (2, 2):
        a = conf.iloc[0, 0]
        b = conf.iloc[0, 1]
        c = conf.iloc[1, 0]
        d = conf.iloc[1, 1]
        return abs((a*d - b*c)/np.sqrt((a + b)*(a + c)*(b + d)*(c + d)))
    chi2 = ss.chi2_contingency(conf)[0]
    n = sum(conf.sum())
    r, k = conf.shape
    m = min(r-1, k-1)
    return np.sqrt(chi2/(m*n))


# def OHE_cat (data, categorial_variables, drop=False):
#     new_categorial_variables = []
#     new_data = pd.DataFrame(data)
#     for variable in categorial_variables:
#         values = np.unique(new_data[variable])
#         for value in values:
#             new_name = variable + ':' + str(value)
#             new_categorial_variables.append(new_name)
#             new_data[new_name] = 0
#         current_names = list(new_data.columns)
#         iter1 = current_names.index(variable)
#         for i in range(new_data.shape[0]):
#             val = new_data.iloc[i, iter1]
#             iter2 = current_names.index(variable + ':' + str(val))
#             new_data.iloc[i, iter2] = 1
#         if drop:
#             new_data.drop(variable, axis=1, inplace=True)
#     return (new_data, new_categorial_variables)


def OHE_cat (data, categorial_variables):
    
    """ data - table, categorial_variables - list of categorial variables to make one-hot-encoding
        returns: new data table with encoded variables, list of new names """
    
    D = []
    for i in categorial_variables:
        df = pd.get_dummies(data[i], prefix=i, prefix_sep=':')
        D.append(df)
    new_data = pd.concat(D, axis=1)
    return (new_data, list(new_data.columns))


def cont_encoding (data, dictionary):
    
    """ data - table, dictionary - dict with keys (names of continious variables)
        and values (lists of correspinding intervals, for example: [-np.inf, 22, 29, 35, 50, np.inf]) """
    
    for variable in dictionary:
        data[variable] = pd.cut(data[variable], dictionary[variable], labels=False, right=True)

        
def get_months (data, DateTime_name):
    
    """ data - table with date column with format <year-month-day> (pd.TimeStamp type), DateTime_name - name of date column """
    
    year_month = [str(i).split(' ')[0][:-3] for i in data[DateTime_name]]
    data['year-month'] = year_month

    
def bins_dynamics_plot (data, var_name, default_name, cumulative=True, criteria='woe', return_data=False):
    
    """ data - table with one-hot-encoded values (with ones and zeros only), var_name - variable name to explore its bins dynamics over months, 
        default_name - name of column with default flag (contains of only ones and zeros), cumulative - whether to use '<=' (True) or '==' (False) for time while plotting graphs, criteria - whether to use 'woe' or 'bad_rate' for analysis """
    
    names = list(data.columns)
    variables = [i for i in names if var_name == i.split(':')[0]]
    months = np.unique(data['year-month'])
    WOE = []
    for variable in variables:
        w = []
        w.append(variable)
        for i in range(len(months)):
            if cumulative == True:
                if criteria == 'woe':
                    w.append(woe(data[data['year-month'] <= months[i]], variable, default_name))
                if criteria == 'bad_rate':
                    w.append(bin_bad_rate(data[data['year-month'] <= months[i]], variable, default_name))
            if cumulative == False:
                if criteria == 'woe':
                    w.append(woe(data[data['year-month'] == months[i]], variable, default_name))
                if criteria == 'bad_rate':
                    w.append(bin_bad_rate(data[data['year-month'] == months[i]], variable, default_name))
        WOE.append(w)
    x = np.sort(list(set(data['year-month'])))
    for i in WOE:
        plt.plot(x, i[1:], label=i[0].split(':')[-1])
        plt.title(var_name)
        plt.ylabel(criteria)
    plt.legend()
    plt.show()
    if return_data:
        dataframe = pd.DataFrame([i[1:] for i in WOE], columns=x, index=[i[0].split(':')[-1] for i in WOE])
        return dataframe.T

def get_corr_matrix (data, filename):
    
    """ create excel file with correlation matrix
    
        data - table with woe values (after fill_data_with_woe), filename - name of excel file to be used"""
    
    list_names = list(data.columns)
    corr_matrix = []
    for i in tqdm(list_names):
        corr_list = []
        for j in list_names:
            conf = pd.crosstab(data[i], data[j])
            correlation = cramer(conf)
            corr_list.append(correlation)
        corr_matrix.append(corr_list)
    pd.DataFrame(data=corr_matrix, index=list_names, columns=list_names).to_excel(filename)


def get_woe_dict (data, variables, default_name):
    
    """ create the following dictionaries: WOE (for using in fill_data_with_woe), WOE1/bad_rate/rate (for creating scorecard)
    
        data - table with ones and zeros, variables - list of variables to create WOE dict (both categorial and continious),
        default_name - name of column with default flag (contains of only ones and zeros)
        
        returns: WOE1, WOE, bad_rate, rate dictionaries """
    
    WOE = {}
    WOE1 = {}
    bad_rate = {}
    rate = {}
    IV = {}
    for variable in variables:
        WOE1[variable], IV[variable] = woe(data, variable, default_name, IV=True)
        rate[variable] = sum(data[variable])/len(data[variable])
        bad_rate[variable] = bin_bad_rate(data, variable, default_name)
        if '//' in variable:
            main = variable.split(':')[0]
            cat = variable.split(':')[1].split('//')
            for value in cat:
                WOE[main + ':' + value] = woe(data, variable, default_name)
        else:
            WOE[variable] = woe(data, variable, default_name)
    return (WOE1, WOE, bad_rate, rate, IV)


def fill_data_with_woe (data, data_to_check, variables, WOE):
    
    """ fill data with corresponding woe values 
    
        data - initial table (with any values), data_to_check - the same table (copy of data), variables - list of names of columns to change, WOE - dict of woe to fill with """
    
    for j in range(len(variables)):
        column = variables[j]
        for i in tqdm(range(data.shape[0])):
            name = column + ':' + str(data_to_check.loc[i, column])
            data.loc[i, column] = WOE[name]