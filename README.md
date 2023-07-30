# Document Analysis with Image Matching and OCR

This repository contains code for performing document analysis using image matching and optical character recognition (OCR). The code aligns two input images and extracts text from specific regions in the aligned image using Tesseract OCR and PaddleOCR.

## Overview

Document analysis is an important task in computer vision that involves understanding and extracting information from images of documents. This repository demonstrates a step-by-step approach to aligning two input images, extracting specific regions of interest (ROI) from the aligned image, and performing OCR to recognize text from those regions.

## How to Use

Follow the steps below to implement the document analysis code:

1. **Install Required Libraries**: Ensure you have the necessary libraries installed to run the code.

2. **Load Input Images**: Prepare two input images that you want to analyze. Update the `image_path1` and `image_path2` variables in the code with the respective paths to your images.

3. **Run the Code**: Execute the code in your preferred Python environment. The code will display the reference and input images side by side, showing the matching keypoints between the images.

4. **Image Matching**: The code uses the SIFT (Scale-Invariant Feature Transform) algorithm for image matching. It detects keypoints and descriptors in both images and matches them using the BFMatcher (Brute-Force Matcher).

5. **Homography Calculation**: The code calculates the homography matrix using the RANSAC algorithm to align the input images.

6. **Image Alignment**: The input image2 is warped using the homography matrix to align it with image1. The aligned image is then displayed.

7. **Text Extraction**: The code extracts text from specific regions of the aligned image. You can specify the regions of interest in the `data` list by providing the coordinates (xmin, xmax, ymin, ymax) of each region and a label for identification.

8. **Text Recognition**: The code uses Tesseract OCR to recognize text from the specified regions of interest. It also demonstrates using PaddleOCR for OCR, which can be an alternative for text recognition.

9. **Result Display**: The extracted text from each region of interest is printed for further analysis.

## Requirements

- Python 3.x
- OpenCV
- Numpy
- Matplotlib
- Pillow
- pytesseract
- tesseract-ocr
- paddlepaddle
- paddleocr

Make sure to have the required libraries installed before running the code.

## Notes

- Ensure that the images are accessible at the specified `image_path1` and `image_path2`.
- The code attempts to find the transformation that aligns `image2` with `image1`, so make sure you have a good-quality image for `image1` which acts as a template image that makes the `image2` (input image) align the perspective of `image1` (template image).
- The SIFT algorithm for image matching and the homography calculation may not work optimally for all types of images. Consider using different feature detection and matching algorithms based on your specific use case.
- Adjust the coordinates in the `data` list to select the regions of interest for text extraction. The OCR accuracy may vary based on the quality and content of the input images.
- Tesseract-OCR is used as an alternative OCR tool, but you can choose to use other OCR libraries or APIs based on your preferences and requirements.

**Disclaimer**: The OCR accuracy heavily depends on the quality and clarity of the input images. Additionally, OCR may not be perfect and may require further post-processing for accurate text recognition.

Please refer to the code comments for more details about the implementation and customization. For any issues or inquiries, feel free to reach out to the repository's owner. Happy document analysis!
