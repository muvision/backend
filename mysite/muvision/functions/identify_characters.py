import classification


def classify_characters(df):
    char_list = []

    for i in range(len(df.index)):
        image = df['image'].iloc[i]
        res = classification.classify(image)
        char_list.append(res)

    return char_list
