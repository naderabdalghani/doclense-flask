from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
from docx import Document
from docx.shared import Inches


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


def output_segmentation_results_imgs(lines):
    i = 0
    for line in lines:
        for word in line.words:
            for character in word.characters:
                io.imsave('./results/' + str(i) + '.png', character.img)
                i += 1


def output_segmentation_results_docx(lines):
    i = 0
    for line in lines:
        for word in line.words:
            for character in word.characters:
                io.imsave('./results/' + str(i) + '.png', character.img)
                i += 1

    # document = Document()

    # document.add_heading('Document Title', 0)
    #
    # p = document.add_paragraph('A plain paragraph having some ')
    # p.add_run('bold').bold = True
    # p.add_run(' and some ')
    # p.add_run('italic.').italic = True
    #
    # document.add_heading('Heading, level 1', level=1)
    # document.add_paragraph('Intense quote', style='Intense Quote')
    #
    # document.add_paragraph(
    #     'first item in unordered list', style='List Bullet'
    # )
    # document.add_paragraph(
    #     'first item in ordered list', style='List Number'
    # )

    # document.add_picture('your_file.jpeg', width=Inches(4))

    # records = (
    #     (3, '101', 'Spam'),
    #     (7, '422', 'Eggs'),
    #     (4, '631', 'Spam, spam, eggs, and spam')
    # )
    #
    # table = document.add_table(rows=1, cols=3)
    # hdr_cells = table.rows[0].cells
    # hdr_cells[0].text = 'Qty'
    # hdr_cells[1].text = 'Id'
    # hdr_cells[2].text = 'Desc'
    # for qty, _id, desc in records:
    #     row_cells = table.add_row().cells
    #     row_cells[0].text = str(qty)
    #     row_cells[1].text = _id
    #     row_cells[2].text = desc

    # document.add_page_break()
    #
    # document_name = os.path.splitext(img_filename)[0] + '.docx'
    # document.save(document_name)
