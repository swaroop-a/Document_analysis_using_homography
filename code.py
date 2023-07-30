# Import the required libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import pytesseract

# Install the required libraries
!pip install pytesseract
!apt-get install tesseract-ocr

def load_images(image_path1, image_path2):
    # Load the two input images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    return image1, image2

def display_images(image1, image2):
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

def preprocess_images(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return gray1, gray2

def detect_and_match_features(image1, image2):
    # Create a feature detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors in the two images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Create a matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match the descriptors
    matches = matcher.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:15], None, flags=2)
    cv2_imshow(matching_result)
    print("Number of matches:", len(matches))

    return keypoints1, keypoints2, descriptors1, descriptors2, matches

def perform_homography(keypoints1, keypoints2, matches, N=50):
    # Select the top N matches
    matches = matches[:N]

    # Extract the matching keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    return homography

def warp_image(image2, homography, shape):
    # Warp image 2 to align with image 1
    warp_image2 = cv2.warpPerspective(image2, homography, shape)
    return warp_image2

def extract_text_from_image(image, data):
    extracted_data = {}
    for crop_data in data:
        label = crop_data['label']
        xmin = crop_data['xmin']
        xmax = crop_data['xmax']
        ymin = crop_data['ymin']
        ymax = crop_data['ymax']

        # Crop the image
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Perform OCR text detection using Tesseract
        text = pytesseract.image_to_string(cropped_image)

        print(label)
        cv2_imshow(cropped_image)
        print(text)


def main():
    # Image paths
    image_path1 = 'reference image path (template image/ perfectly alligned image)'
    image_path2 = 'input image path (image for which we need to extract text)'

    # Load the images
    image1, image2 = load_images(image_path1, image_path2)

    # Display the images
    display_images(image1, image2)

    # Preprocess the images
    gray1, gray2 = preprocess_images(image1, image2)

    # Detect and match features
    keypoints1, keypoints2, descriptors1, descriptors2, matches = detect_and_match_features(gray1, gray2)

    # Perform homography
    homography = perform_homography(keypoints1, keypoints2, matches)

    # Warp image
    warp_image2 = warp_image(image2, homography, image1.shape[1::-1])

    print("####### OUTPUT ######")

    # Combine the warped image and image 1
    result = np.concatenate((image1, warp_image2), axis=1)

    # Display the result
    cv2_imshow(result)

    # Extract text from warped image using custom data
    # Change the labels and co-ordinates according to your usecase and image 
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
        {
            'label': 'date of birth',
            'xmin': 79,
            'xmax': 269,
            'ymin': 59,
            'ymax': 85
        },
        {
            'label': 'gender',
            'xmin': 78,
            'xmax': 204,
            'ymin': 74,
            'ymax': 100
        },
        {
            'label': 'aadhar number',
            'xmin': 69,
            'xmax': 206,
            'ymin': 122,
            'ymax': 157
        }
    ]
    extract_text_from_image(warp_image2, data)

    
if __name__ == "__main__":
    main()
