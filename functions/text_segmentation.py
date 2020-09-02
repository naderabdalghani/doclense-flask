import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from matplotlib import pyplot as plt
from skimage import img_as_bool
from skimage.filters import median
from skimage.morphology import binary_erosion
from skimage.transform import resize, rescale
from skimage.io import imsave
from functions.thresholding import binarize


def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


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


def text_to_lines(binary):  # this function takes a binary image
    # and returns array of line segment objects
    # which contains (for every line) the image of the line + starting row and ending row
    # in the original image
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
    # print(line_histogram_thresh)
    while k in range(len(line_histogram)):
        if line_histogram[k] >= line_histogram_thresh:
            # print(line_histogram[k])
            start = k
            for end in range(start, len(line_histogram)):
                # print("end is ",end)
                if line_histogram[end] <= line_histogram_thresh:
                    line_segments.append(LineSegment(start, end))
                    k = end
                    # print("inner k =",k)
                    break
            if end >= len(line_histogram) - 1:
                break

        k += 1

    # print(line_segments)
    # dialated_lines=[] #for word extraction
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
            # plt.imshow(binary)
            line_segments[g].img = binary[line_segments[g].start_row - 5:line_segments[g].end_row + 5, 0:width]
            line_segments[g].img = median(line_segments[g].img)
            # image_from_arr = Image.fromarray(line_segments[g].img)
            # image_from_arr.save("line " + str(count_lines) + " " + '.png')
            n = np.copy(line_segments[g].img)
            d = 1 - (ndimage.binary_dilation(1 - n, iterations=round(width / 319)))

            # c=np.invert(binary_closing(np.invert(i.img)))
            # c=np.invert(binary_closing(np.invert(c)))
            # c=np.invert(binary_closing(np.invert(c)))
            # image_from_arr=Image.fromarray(d)
            # image_from_arr.save("linedialated "+str(count_lines)+" "+'.png')
            # image_from_arr=Image.fromarray(c)
            # image_from_arr.save("lineclosed "+str(count_lines)+" "+'.png')
            line_segments[g].img_dialated = d
            # image_from_arr = Image.fromarray(d)
            # image_from_arr.save("linedialated " + str(count_lines) + " " + '.png')
            count_lines += 1
        else:
            del line_segments[g]
            g = g - 1
        g = g + 1
    return line_segments
    # plt.plot(line_histogram)
    # plt.show()
    # io.imshow(binary)
    # image_from_arr=Image.fromarray(binary)
    # image_from_arr.save('binarized.png')


def lines_to_words(line_segments):  # this function takes array of lines and adds
    # the following (for each line ):
    # array of word segment objects which contains (for each word):
    # word image + start column + end column
    count_lines = 0
    for line in line_segments:
        count_words = 0
        # show_images([l.img])
        if line.end_row - line.start_row >= 15:
            height = line.img_dialated.shape[0]
            width = line.img_dialated.shape[1]

            word_histogram = np.zeros(width)
            for j in range(width):
                for i in range(height):
                    if line.img_dialated[i][j] == 0:
                        word_histogram[j] += 1

            # print("histogram fpr words @ line "+str(count_lines)+" : ",word_histogram)
            k = 0
            line.words = []  # set the variable 3shan kant btdrb
            # print(word_histogram)
            word_histogram_thresh = round(np.average(word_histogram))
            word_histogram_thresh /= 1.5
            # print("threshold = ", word_histogram_thresh)
            while k in range(len(word_histogram)):
                if word_histogram[k] >= word_histogram_thresh:
                    # print("1st k=",k)
                    for end in range(k, len(word_histogram) - 2):
                        if word_histogram[end] < 1 and word_histogram[end + 1] < 1 and word_histogram[end + 2] < 1:
                            line.words.append(WordSegment(k, end))
                            k = end
                            # print("last k=",k)
                            break

                k += 1
            # words_test=[]
            for m in line.words:
                # print(m)
                # print(m.start_col, m.end_col)
                x = line.img[0:height, m.start_col:m.end_col]
                m.word_img = x
                # image_from_arr=Image.fromarray(x)
                # image_from_arr.save(str(count_lines)+"-"+str(count_words)+'.png')
                # words_test.append(binary_dilation(x))
                # show_images([binary_dilation(x)])
                count_words += 1
            count_lines += 1


# this function takes array of lines and
# adds the following (for each line):
#    for each word:
#       array of char objects which contains :
# image of the character + starting column + end column
def words_to_characters(lines):
    count_lines = 0
    count_words = 0
    count_char = 0
    # show_images([line.img])
    for line in range(len(lines)):  # loop on every line
        if lines[line].end_row - lines[line].start_row >= 15:
            for w in range(len(lines[line].words)):  # loop on every word in the line
                height_w = lines[line].words[w].word_img.shape[0]
                width_w = lines[line].words[w].word_img.shape[1]
                char_histogram = np.zeros(width_w)  # vertical histogram for the word
                count_char = 0
                for j in range(width_w):
                    for i in range(height_w):
                        if lines[line].words[w].word_img[i][j] == 0:
                            char_histogram[j] += 1

                k = 0
                end = -1
                lines[line].words[w].characters = []  # set the variable to prevent error
                # print(char_histogram)
                char_histogram_thresh = round(np.average(char_histogram))
                char_histogram_thresh /= 2
                char_histogram_thresh = round(char_histogram_thresh)
                while k in range(len(char_histogram)):  # loop on the histogram
                    if char_histogram[k] > 0:  # k is the first column to have a black pixel,so we save it
                        # print("1st k=", k)
                        for end in range(k, len(char_histogram) - 1):  # loop starts with k till the end
                            if ((char_histogram[end] == 0) and (
                                    char_histogram[end + 1] == 0)):  # if we found an empty area,the char is completed
                                lines[line].words[w].characters.append(
                                    Character(k, end))  # create an object with the coordinates
                                k = end
                                # print("last k=", k)
                                break
                        if end + 1 == len(char_histogram) - 1:
                            lines[line].words[w].characters.append(
                                Character(k, end))  # create an object with the coordinates
                            k = end
                            # print("last k=", k)
                            break

                    k += 1
                for m in range(
                        len(lines[line].words[w].characters)):  # loop on these coordinates to cut the word into chars
                    # print(m)
                    # print(m.start_col, m.end_col)
                    if lines[line].words[w].characters[m].end_col - lines[line].words[w].characters[m].start_col >= 3:
                        x = lines[line].words[w].word_img[0:height_w,
                            lines[line].words[w].characters[m].start_col:lines[line].words[w].characters[
                                m].end_col]  # chop! chop!
                        # image_from_arr=Image.fromarray(x)
                        # image_from_arr.save('line '+str(count_lines)+"- word  "+str(count_words)+" char before "+str(count_char)+'.png')#saving the image
                        # words_test.append(binary_dilation(x))
                        # show_images([x])
                        # resizing the image into 128 x 128
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
                        # image_from_arr = Image.fromarray(x)
                        # image_from_arr.save('line '+str(count_lines)+"- word  "+str(count_words)+" char "+str(count_char)+'.png')
                        # show_images([x])
                        # if height >= width:
                        #     factor_height = int(108 / height)
                        #     new_width = int(width * factor_height)
                        #     if new_width % 2 != 0:
                        #         new_width += 1
                        #     resized = resize(x, (118, new_width), anti_aliasing=True)
                        #     window = np.ones((128, 128))
                        #     window[5:123, (64 - int(new_width / 2)):(64 + int(new_width / 2))] = resized
                        #     # window=np.logical_and(window==1 , resized)
                        # else:
                        #     factor_width = int(108 / width)
                        #     new_height = int(height * factor_width)
                        #     if new_height % 2 != 0:
                        #         new_height += 1
                        #     resized = resize(x, (new_height, 118), anti_aliasing=True)
                        #     window = np.ones((128, 128))
                        #     window[(64 - int(new_height / 2)):(64 + int(new_height / 2)), 5:123] = resized
                        # out_size = (128, 128)
                        # show_images([x])
                        desired_size = 64
                        while x.shape[0] > desired_size * 0.8 or x.shape[1] > desired_size * 0.8:
                            x = rescale(x, 0.9, preserve_range=True)
                            x = binarize(x, mode=2)
                        extra_left = int(np.ceil((desired_size - x.shape[0]) / 2))
                        extra_right = int(desired_size - x.shape[0] - extra_left)
                        extra_bottom = int(np.ceil((desired_size - x.shape[1]) / 2))
                        extra_top = int(desired_size - x.shape[1] - extra_bottom)
                        resized_img = np.pad(x, ((extra_left, extra_right), (extra_top, extra_bottom)), mode='constant', constant_values=1)
                        # show_images([x, resized_img])

                        # show_images([resized_img])
                        # show_images([resized])
                        # show_images([resized_no_bool])
                        # image_from_arr = Image.fromarray(window)
                        # image_from_arr.convert('P')
                        # image_from_arr.save(
                        #     './results/line ' + str(count_lines) + "- word  " + str(
                        #         count_words) + " char resized " + str(
                        #         count_char) + '.png')
                        lines[line].words[w].characters[m].img = resized_img
                        count_char += 1
                count_words += 1
            count_lines += 1
            count_words = 0
