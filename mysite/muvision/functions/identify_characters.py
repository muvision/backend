from . import classification
from . import special_characters as sc
import numpy as np
import os
import tensorflow as tf


def identify_lines(line_info):
    document = "\\begin{align*} "
    for i in range(len(line_info)):
        avg_x = (np.average(line_info[i]['x'].diff()) + np.average(line_info[i]['x1'].diff()))/2
        line_string = classify_characters(line_info[i], avg_x)
        document = document + line_string

        if i != len(line_info) - 1:
            document = document + " \\\\ "
    document = document + " \\end{align*}"
    return document


def classify_characters(df, avg_x):
    line_string = ""
    is_equal = False
    is_script = "reg"
    custom_model_path = os.path.join(os.getcwd(), 'muvision', 'custom_model2.h5')
    custom_model = tf.keras.models.load_model(custom_model_path)

    for i in range(len(df.index)):
        if is_equal:
            is_equal = False
            continue
        image = df['image'].iloc[i]
        res = classification.classify(image, custom_model)

        if res == '-' and sc.determine_equal(i, df, avg_x):

            if is_script == "reg":
                line_string = line_string + " &= "
            else:
                line_string = line_string + " = "
            is_equal = True
        else:
            line_string, is_script = fix_subscripts(line_string, df, is_script, i)
            line_string = line_string + res + " "

    if is_script != 'reg':
        line_string = line_string + "}"

    return line_string


def fix_subscripts(line_string, df, is_script, i):
    special_char = df['character type'].iloc[i]
    if special_char == 'sup' and is_script == "reg":
        line_string = line_string + '^{'
        is_script = 'sup'
    elif special_char == 'sub' and is_script == "reg":
        line_string = line_string + '_{'
        is_script = 'sub'
    elif special_char != is_script:
        line_string = line_string + '}'
        is_script = special_char
    return line_string, is_script
