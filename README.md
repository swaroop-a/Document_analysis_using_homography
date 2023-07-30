# Document_analysis_using_homography
IIT Delhi internship project

# Document Image Alignment and Text Extraction

This repository contains code for aligning two document images and extracting text from the aligned image. The alignment is achieved using feature detection and matching, and the text extraction is performed using both Tesseract OCR and PaddleOCR.

## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites installed in your environment:

- Python 3.6 or higher
- OpenCV (`cv2`) library
- NumPy library
- Matplotlib library
- Tesseract OCR (`pytesseract`) library
- PaddlePaddle library
- PaddleOCR library

### Installation

To install the required libraries, run the following commands:

```bash
pip install opencv-python numpy matplotlib pytesseract paddlepaddle paddleocr
apt-get install tesseract-ocr
```

### Usage

Follow the steps below to implement the document image alignment and text extraction:

1. Import the necessary libraries:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
```

2. Load the two input images:

```python
image_path1 = '/path/to/image1.jpg'
image_path2 = '/path/to/image2.jpg'

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)
```

3. Display the input images:

```python
# Convert the images from BGR to RGB for proper display with matplotlib
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Display the images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Set the title and display image1
ax1.set_title('Reference Image')
ax1.imshow(image1_rgb)
ax1.axis('off')

# Set the title and display image2
ax2.set_title('Input Image')
ax2.imshow(image2_rgb)
ax2.axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
```

4. Preprocess the images and perform feature detection and matching:

```python
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create a feature detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the two images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create a matcher
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors
matches = matcher.match(descriptors1, descriptors2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)
```

5. Perform homography to align the two images:

```python
# Select the top N matches
N = 50
matches = matches[:N]

# Extract the matching keypoints from both images
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix
homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

# Warp image 2 to align with image 1
warp_image2 = cv2.warpPerspective(image2, homography, (image1.shape[1], image1.shape[0]))
```

6. Extract text from the aligned image using custom data:

```python
data = [
    {
        'label': 'name hindi',
        'xmin': 79,
        'xmax': 267,
        'ymin': 30,
        'ymax': 58
    },
    {
        'label': 'name english',
        'xmin': 78,
        'xmax': 272,
        'ymin': 46,
        'ymax': 71
    },
    # Add more crop data as needed
]

# Extract text using Tesseract OCR
extracted_data = {}
for crop_data in data:
    label = crop_data['label']
    xmin = crop_data['xmin']
    xmax = crop_data['xmax']
    ymin = crop_data['ymin']
    ymax = crop_data['ymax']

    # Crop the image
    cropped_image = warp_image2[ymin:ymax, xmin:xmax]

    # Perform OCR text detection using Tesseract
    text = pytesseract.image_to_string(cropped_image)

    extracted_data[label] = text.strip()
```

7. Perform OCR using PaddleOCR:

```python
ocr = PaddleOCR()

# Perform text detection and recognition
result = ocr.ocr(warp_image2)

# Extract recognized text
recognized_text = '\n'.join([line[1][0] for line in result[0]])
```

8. Display the aligned image and extracted text:

```python
# Combine the warped image and image 1
result = np.concatenate((image1, warp_image2), axis=1)

# Display the result
cv2.imshow(result)
```

### Acknowledgments

The code in this repository is based on various OpenCV, NumPy, Matplotlib, Tesseract OCR, PaddlePaddle, and PaddleOCR documentation and tutorials. Special thanks to the developers of these libraries for providing such useful tools for image processing and OCR tasks.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
