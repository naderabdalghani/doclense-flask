<br />
<p align="center">
  <a href="https://github.com/naderabdalghani/doclense-flask">
    <img src="static/images/logo_with_text_solid.png" alt="Logo" width="252" height="72">
  </a>

  <p align="center">
    An app for extracting text from camera-captured photos into .docx files
  </p>
</p>

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Running](#running)
* [Usage](#usage)
* [Results](#results)
* [Roadmap](#roadmap)
* [Contributors](#contributors)
* [Acknowledgements](#acknowledgements)

## About The Project

![App Showcase][product-screenshot]

### Built With

* [Flask](http://flask.palletsprojects.com/en/1.1.x/)
* [PyTorch](https://pytorch.org/)
* This app uses a slightly modified version of this [Kaggle kernel](https://www.kaggle.com/naderabdalghani/printed-letters-classifier/) as its letters classifier model. This model uses a pre-trained Wide ResNet-50-2 convolutional deep neural network which achieves 99% accuracy after training it on this [dataset](https://www.kaggle.com/naderabdalghani/camerataken-images-of-printed-english-alphabet).

## Getting Started

### Prerequisites

* Setup Python using this [link](https://realpython.com/installing-python/)

### Installation

1. Create a virtual environment
	`cd <project-directory>`
	- On Unix-based OS's:
	`$ python3 -m venv venv`
	- On Windows:
	`> py -3 -m venv venv`

2. Activate the environment
	- On Unix-based OS's:
	`$ . venv/bin/activate`
	- On Windows:
	`> venv\Scripts\activate`

3. Install app dependencies
	`pip install -r requirements.txt`

4. Create the following directories in the project main directory
	- `<project-directory>\results`
	- `<project-directory>\uploads`

5. Download the trained model [state dictionary](https://www.kaggle.com/naderabdalghani/printed-letters-classifier/output) and place it in `<project-directory>\model\`

6. **[Optional]** Download the [dataset](https://www.kaggle.com/naderabdalghani/camerataken-images-of-printed-english-alphabet) used and extract it in `<project-directory>\model\`

### Running

* Make sure you are in the project directory
	`cd <project-directory>`
	- On Unix-based OS's:
	`$ python api.py`
	- On Windows:
	`> py -3 api.py`
	

## Usage

Simply click on the 'Upload' button and select a photo that contains printed text. Click on the 'Submit' button and wait briefly for your .docx file to start downloading.

## Results

### Test Case 0

![test_0][test-0]

### Test Case 1

![test_1][test-1]

## Roadmap

### List of Proposed Improvements

* Fix and integrate the [de-skewing script](functions/deskew.py)
* Make the app more tolerant to closely-spaced words
* Train the model on letters with different fonts
* Copy the indentation and format of the printed sheet
* Error handling of missing directories

## Contributors

- [Nader AbdAlGhani](https://github.com/naderabdalghani)
	- [Text separation](functions/text_separation.py) implementation
	- Dataset creation
	- [Classifier model](model/model.py) implementation
	- [Utility functions](functions/utils.py) implementation
	- Web app development and integration
- [Mohamad Ahmad](https://github.com/MouhamedAhmed)
	- Lots of research
	- [De-skewing algorithm](functions/deskew.py) implementation
- [Mostafa Walid](https://github.com/sha3er97)
	- [Thresholding algorithms](functions/thresholding.py) implementation
- [Omar Salah](https://github.com/arminArlert997)
	- [Segmenting](functions/text_segmentation.py) input pages into lines, lines into words and words into letters

## Acknowledgements

* [Dribbble](https://dribbble.com/shots/5489323-doclense-photo-scanner-logo-design)
* [One Page Love](https://onepagelove.com/leno)
* [Random Paragraph Generator](https://randomwordgenerator.com/paragraph.php)

<!-- MARKDOWN LINKS & IMAGES -->

[product-screenshot]: static/images/app-showcase.png
[test-0]: static/images/test_0.png
[test-1]: static/images/test_1.png
