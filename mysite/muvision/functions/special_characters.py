import numpy as np


# Determining Subscripts and superscripts

def determine_special_functions(df):
    std = np.std(df['median_y'])
    y1mean = np.average(df['y1'])
    ymean = np.average(df['y'])
    median_mean = np.average(df['median_y'])

    spec_list = []

    for row in df.index:
        if df['median_y'].iloc[row] < median_mean + std and df['y1'].iloc[row] < median_mean:
            spec_list.append('sup')
        elif df['median_y'].iloc[row] > median_mean - std and df['y'].iloc[row] > median_mean:
            spec_list.append('sub')
        else:
            spec_list.append('reg')

    return spec_list


def determine_equal():
    print("temp")
