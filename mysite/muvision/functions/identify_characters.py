import classification
import special_characters as sc
import numpy as np


def identify_lines(line_info):
    document = []
    for df in line_info:
        avg_x = (np.average(df['x'].diff()) + np.average(df['x1'].diff()))/2
        line_string = classify_characters(df, avg_x)
        document.append(line_string)
    return document


def classify_characters(df, avg_x):
    line_string = ""
    is_equal = False
    is_script = "reg"

    for i in range(len(df.index)):
        if is_equal:
            is_equal = False
            continue
        image = df['image'].iloc[i]
        res = classification.classify(image)

        if res == '-' and sc.determine_equal(i, df, avg_x):
            line_string, is_script = fix_subscripts(line_string, df, is_script, i)
            line_string = line_string + "= "
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
