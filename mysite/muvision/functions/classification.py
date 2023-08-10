import tensorflow as tf
import numpy as np
import os
from PIL import Image
from skimage import transform
from skimage import morphology
from skimage import util
from skimage.filters import threshold_otsu

# This one uses skeletonization which works better for thicker drawings
def classify(np_image, model):
  try:
    np_image = np.array(np_image).astype('uint8')/255

    # invert black and white because skeletonization wants white drawing on black bg
    np_image = util.invert(np_image)
    np_image = transform.resize(np_image, (45, 45,3))

    # threshold so that there's no gray
    thresh = threshold_otsu(np_image)
    np_image = np_image > thresh

    # skeletonize the image, meaning make the black part as skinny as 1 px
    np_image = morphology.skeletonize(np_image)
    np_image = util.invert(np_image)

    white_pixels_mask = np.all(np_image == [255, 255, 255], axis=-1)
    non_white_pixels_mask = ~white_pixels_mask

    np_image[white_pixels_mask] = [255, 255, 255]
    np_image[non_white_pixels_mask] = [0,0,0]
    np_image = np.array(np_image).astype('uint8')/255

    np_image = np.expand_dims(np_image, axis=0)
    prediction = model.predict(np_image)

    class_order = ['!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'M', 'N', 'S', 'T', 'X', '[', ']', 'b', 'd', 'e', 'f', 'i', 'l', 'o', 'p', 'q', 'u', 'v', 'y', 'z', '{', '}']

    tensor = tf.math.argmax(prediction[0])
    return class_order[int(tensor)]
  except:
    print("error")
    return 'L'


# old classify function
# def classify(np_image, model):
#     # np_image = Image.open(filename)
#     # print(shape(np_image))
#     #  np_image = tf.image.rgb_to_grayscale(np_image, name='jeff')
#     # np_image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
#     np_image = np.array(np_image).astype('float32')/256
#     np_image = transform.resize(np_image, (45, 45, 3))
#     np_image = np.expand_dims(np_image, axis=0)
#     prediction = model.predict(np_image)
#     class_order = ['!', '(', ')', '+', '/', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', '\\Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', '\\alpha', '|', 'b', '\\beta', '\\cos', 'd', 'div', 'e', '\\exists', 'f', '\\forall', '\\mathbin{/}', 'gamma', '\\geq', '\\gt', 'i', '\\in', '\\infty', '\\int', 'j', 'k', 'l', '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', 'o', 'p', '\\phi', '\\pi', '\\pm', '\\prime', 'q', '\\rightarrow', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', 'u', 'v', 'w', 'y', 'z', '\\{', '\\}']
#     tensor = tf.math.argmax(prediction[0])
#     return class_order[int(tensor)]

