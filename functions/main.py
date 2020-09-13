import os
import skimage.io as io
from skimage.color import rgb2gray
from functions.text_separation import text_separation
from functions.thresholding import binarize
from functions.text_segmentation import text_to_lines, lines_to_words, words_to_characters
from functions.utils import output_segmentation_results_docx
from model.model import *


def main(img_filename):
    img_file_path = os.path.join(os.path.dirname(__file__), '../uploads/' + img_filename)
    img = io.imread(img_file_path)
    img = rgb2gray(img) * 255
    img = text_separation(img)
    img = binarize(img, mode=2)
    lines = text_to_lines(img)
    lines_to_words(lines)
    words_to_characters(lines)
    model = init_model()
    load_trained_model(model)
    class_names = get_class_names()
    output_segmentation_results_docx(lines, model, class_names)
    return
