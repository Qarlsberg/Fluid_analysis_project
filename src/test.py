import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def process_image(image_path):
    try:
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Apply Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 4: K-means segmentation
        # Reshape image for k-means
        pixel_values = blurred.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        # Define criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = 3
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(gray.shape)
        
        # Display results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(221)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(222)
        plt.title('Grayscale')
        plt.imshow(gray, cmap='gray')
        plt.axis('off')
        
        plt.subplot(223)
        plt.title("Otsu's Threshold")
        plt.imshow(otsu, cmap='gray')
        plt.axis('off')
        
        plt.subplot(224)
        plt.title('K-means Segmentation (k=3)')
        plt.imshow(segmented_image, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    # Use the correct path to Example.jpg
    image_path = os.path.join(os.path.dirname(__file__), 'Example.jpg')
    process_image(image_path)
