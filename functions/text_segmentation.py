import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import median
from skimage.morphology import binary_erosion
from skimage.transform import rescale

from functions.thresholding import binarize


class LineSegment:
    def __init__(self, start_row, end_row, img_dialated=0):
        self.start_row = start_row
        self.end_row = end_row
        self.words = []
        self.img = []
        self.img_dialated = []


class WordSegment:
    def __init__(self, start_col, end_col):
        self.start_col = start_col
        self.end_col = end_col
        self.word_img = []
        self.characters = []


class Character:
    def __init__(self, start_col=0, end_col=0):
        self.start_col = start_col
        self.end_col = end_col
        self.img = []


def text_to_lines(binary):
    height = binary.shape[0]
    width = binary.shape[1]
    b = np.copy(binary)
    dialated_binary = binary_erosion(b)
    line_histogram = np.zeros(height)
    for j in range(width):
        for i in range(height):
            if dialated_binary[i][j] == 0:
                line_histogram[i] += 1
    line_segments = []
    k = 0
    start = 0
    end = 0
    line_histogram_thresh = np.average(line_histogram)
    line_histogram_thresh /= 10
    while k in range(len(line_histogram)):
        if line_histogram[k] >= line_histogram_thresh:
            start = k
            for end in range(start, len(line_histogram)):
                if line_histogram[end] <= line_histogram_thresh:
                    line_segments.append(LineSegment(start, end))
                    k = end
                    break
            if end >= len(line_histogram) - 1:
                break

        k += 1

    count_lines = 0
    g = 0
    line_thresh = 0
    heights = []
    for line in line_segments:
        heights.append(line.end_row - line.start_row)
    line_thresh = np.mean(heights)
    lower_bound = line_thresh - 0.25 * line_thresh
    upper_bound = line_thresh + 0.25 * line_thresh
    while g < len(line_segments):
        if lower_bound <= line_segments[g].end_row - line_segments[g].start_row <= upper_bound:
            line_segments[g].img = binary[line_segments[g].start_row - 5:line_segments[g].end_row + 5, 0:width]
            line_segments[g].img = median(line_segments[g].img)
            n = np.copy(line_segments[g].img)
            d = 1 - (ndimage.binary_dilation(1 - n, iterations=round(width / 319)))
            line_segments[g].img_dialated = d
            count_lines += 1
        else:
            del line_segments[g]
            g = g - 1
        g = g + 1
    return line_segments


def lines_to_words(line_segments):
    count_lines = 0
    for line in line_segments:
        count_words = 0
        if line.end_row - line.start_row >= 15:
            height = line.img_dialated.shape[0]
            width = line.img_dialated.shape[1]

            word_histogram = np.zeros(width)
            for j in range(width):
                for i in range(height):
                    if line.img_dialated[i][j] == 0:
                        word_histogram[j] += 1

            k = 0
            line.words = []
            word_histogram_thresh = round(np.average(word_histogram))
            word_histogram_thresh /= 1.5
            while k in range(len(word_histogram)):
                if word_histogram[k] >= word_histogram_thresh:
                    for end in range(k, len(word_histogram) - 2):
                        if word_histogram[end] < 1 and word_histogram[end + 1] < 1 and word_histogram[end + 2] < 1:
                            line.words.append(WordSegment(k, end))
                            k = end
                            break

                k += 1
            for m in line.words:
                x = line.img[0:height, m.start_col:m.end_col]
                m.word_img = x
                count_words += 1
            count_lines += 1


def words_to_characters(lines):
    count_lines = 0
    count_words = 0
    count_char = 0
    for line in range(len(lines)):
        if lines[line].end_row - lines[line].start_row >= 15:
            for w in range(len(lines[line].words)):
                height_w = lines[line].words[w].word_img.shape[0]
                width_w = lines[line].words[w].word_img.shape[1]
                char_histogram = np.zeros(width_w)
                count_char = 0
                for j in range(width_w):
                    for i in range(height_w):
                        if lines[line].words[w].word_img[i][j] == 0:
                            char_histogram[j] += 1

                k = 0
                end = -1
                lines[line].words[w].characters = []
                char_histogram_thresh = round(np.average(char_histogram))
                char_histogram_thresh /= 2
                char_histogram_thresh = round(char_histogram_thresh)
                while k in range(len(char_histogram)):
                    if char_histogram[k] > 0:
                        for end in range(k, len(char_histogram) - 1):
                            if ((char_histogram[end] == 0) and (
                                    char_histogram[end + 1] == 0)):
                                lines[line].words[w].characters.append(
                                    Character(k, end))
                                k = end
                                break
                        if end + 1 == len(char_histogram) - 1:
                            lines[line].words[w].characters.append(
                                Character(k, end))
                            k = end
                            break

                    k += 1
                for m in range(
                        len(lines[line].words[w].characters)):
                    if lines[line].words[w].characters[m].end_col - lines[line].words[w].characters[m].start_col >= 3:
                        x = lines[line].words[w].word_img[0:height_w,
                            lines[line].words[w].characters[m].start_col:lines[line].words[w].characters[
                                m].end_col]  # chop! chop!
                        height = x.shape[0]
                        width = x.shape[1]
                        y_start = -1
                        y_end = -1
                        x_start = -1
                        x_end = -1
                        for i in range(height):
                            for j in range(width):
                                if x[i][j] == 0:
                                    y_start = i
                                    break
                                else:
                                    continue
                            if y_start != -1:
                                break
                        for i in range(height):
                            for j in range(width):
                                if x[i][j] == 0:
                                    y_end = i
                        for j in range(width):
                            for i in range(height):
                                if x[i][j] == 0:
                                    x_start = j
                                    break
                                else:
                                    continue
                            if x_start != -1:
                                break
                        for j in range(width):
                            for i in range(height):
                                if x[i][j] == 0:
                                    x_end = j
                        if y_start > 3 and y_end < height - 2 and x_start > 2 and x_end < width - 2:
                            x = x[y_start - 3:y_end + 2, x_start - 2:x_end + 2]
                        elif y_start > 3 and y_end < height - 2:
                            x = x[y_start - 3:y_end + 2, x_start:x_end]
                        elif x_start > 2 and x_end < width - 2:
                            x = x[y_start:y_end, x_start - 2:x_end + 2]

                        height = x.shape[0]
                        width = x.shape[1]
                        desired_size = 64
                        while x.shape[0] > desired_size * 0.8 or x.shape[1] > desired_size * 0.8:
                            x = rescale(x, 0.9, preserve_range=True)
                            # x = binarize(x, mode=2)
                        extra_left = int(np.ceil((desired_size - x.shape[0]) / 2))
                        extra_right = int(desired_size - x.shape[0] - extra_left)
                        extra_bottom = int(np.ceil((desired_size - x.shape[1]) / 2))
                        extra_top = int(desired_size - x.shape[1] - extra_bottom)
                        resized_img = np.pad(x, ((extra_left, extra_right), (extra_top, extra_bottom)), mode='constant',
                                             constant_values=1)
                        lines[line].words[w].characters[m].img = resized_img
                        count_char += 1
                count_words += 1
            count_lines += 1
            count_words = 0
