import numpy as np
from skimage.morphology import reconstruction, dilation


def background_elimination(img):
    img_copy = np.copy(img)
    block_size = 30
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i + block_size, j:j + block_size]
            gmin = np.amin(block)
            gmax = np.amax(block)
            tfixed = 100
            tmin = 50
            tvar = ((gmin - tmin) - min(tfixed, gmin - tmin)) * 2
            t = tvar + tfixed
            intensity_variance = gmax - gmin
            if intensity_variance < t:
                img_copy[i:i + block_size, j:j + block_size] = 255
    return img_copy


def graphics_separation(img):
    structural_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    img_copy = np.copy(img)

    for i in range(10):
        img_copy = dilation(img_copy, structural_element)

    for i in range(10):
        img_copy = reconstruction(img_copy, img, 'erosion', structural_element)

    return img_copy


def text_separation(img):
    background_eliminated_img = background_elimination(img)
    graphical_content_img = graphics_separation(background_eliminated_img)
    textual_content_img = background_eliminated_img - graphical_content_img
    return textual_content_img
