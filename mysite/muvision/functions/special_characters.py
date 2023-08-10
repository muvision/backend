import numpy as np
from . import classification
import os
import tensorflow as tf

# Determining Subscripts and superscripts


def determine_special_functions(df):
    std = np.std(df['median_y'])
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


def determine_equal(i, df, avg_x):
    custom_model_path = os.path.join(os.getcwd(), 'muvision', 'custom_model2.h5')
    custom_model = tf.keras.models.load_model(custom_model_path)

    if i + 1 < len(df.index):
        cur_median = (df['x1'].iloc[i] - df['x'].iloc[i])/2
        next_median = (df['x1'].iloc[i+1] - df['x'].iloc[i+1])/2
        next_symbol = classification.classify(df['image'].iloc[i + 1], custom_model)
        #and abs(next_median - cur_median) < avg_x
        if next_symbol == '-':
            return True
    return False
