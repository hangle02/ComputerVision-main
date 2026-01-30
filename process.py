import cv2
import numpy as np
import os

class ImageProcessor:
    """
    Class for processing images from camera feed.
    Implements 10 computer vision filtering techniques.
    """
    
    def __init__(self):
        pass
    
    # =============================================================================
    # FILTER IMPLEMENTATIONS
    # =============================================================================
    
    def convert_to_grayscale(self, bgr_img):
        """1. Grayscale Conversion"""
        if bgr_img is None: return None
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, img, kernel_size=(5, 5)):
        """2. Gaussian Blur"""
        if img is None: return None
        return cv2.GaussianBlur(img, kernel_size, 0)

    def apply_median_blur(self, img, kernel_size=5):
        """3. Median Blur"""
        if img is None: return None
        # Kernel size must be odd
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(img, k)

    def apply_sobel_x(self, img):
        """4. Sobel Edge Detection (X Direction)"""
        if img is None: return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # cv2.CV_64F supports negative numbers (derivatives)
        sobelx_64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        return cv2.convertScaleAbs(sobelx_64f)

    def apply_laplacian(self, img):
        """5. Laplacian Edge Detection"""
        if img is None: return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        laplacian_64f = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian_64f)

    def apply_sharpening(self, img):
        """6. Sharpening Filter"""
        if img is None: return None
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def apply_bilateral_filter(self, img, d=9, sigma_color=75, sigma_space=75):
        """7. Bilateral Filter"""
        if img is None: return None
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def apply_threshold(self, img, threshold_value=127):
        """8. Thresholding (Binary)"""
        if img is None: return None
        gray_img = img
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
        return binary

    def apply_morphology(self, binary_img, operation='close', kernel_size=(5, 5)):
        """9 & 10. Morphological Filtering (Erosion & Dilation)"""
        if binary_img is None: return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        if operation == 'erode':
            return cv2.erode(binary_img, kernel, iterations=1)
        elif operation == 'dilate':
            return cv2.dilate(binary_img, kernel, iterations=1)
        
        return binary_img

    # =============================================================================
    # MAIN PROCESSING PIPELINE
    # =============================================================================
    
    # !!! THIS IS THE FUNCTION THAT CAUSED THE ERROR !!!
    # It must accept the 'step' argument.
    def process_frame(self, frame, step='all'):
        if frame is None: return None, {}, 0
        
        start_time = cv2.getTickCount()
        results = {}
        processed_img = frame.copy()
        
        # --- Logic to switch filters based on 'step' ---
        if step == 'gray':
            processed_img = self.convert_to_grayscale(frame)
            
        elif step == 'gaussian':
            processed_img = self.apply_gaussian_blur(frame)
            
        elif step == 'median':
            processed_img = self.apply_median_blur(frame)
            
        elif step == 'sobel':
            processed_img = self.apply_sobel_x(frame)
            
        elif step == 'laplacian':
            processed_img = self.apply_laplacian(frame)
            
        elif step == 'sharpen':
            processed_img = self.apply_sharpening(frame)
            
        elif step == 'bilateral':
            processed_img = self.apply_bilateral_filter(frame)
            
        elif step == 'threshold':
            processed_img = self.apply_threshold(frame)
            
        elif step == 'erode':
            # Erosion needs a binary image first
            binary = self.apply_threshold(frame)
            processed_img = self.apply_morphology(binary, operation='erode')
            
        elif step == 'dilate':
            # Dilation needs a binary image first
            binary = self.apply_threshold(frame)
            processed_img = self.apply_morphology(binary, operation='dilate')
            
        elif step == 'all':
            # Default behavior (e.g., just show original or a specific demo)
            results['info'] = "No specific filter selected"
            pass

        # Calculate processing time (ms)
        end_time = cv2.getTickCount()
        time_ms = (end_time - start_time) * 1000 / cv2.getTickFrequency()
        
        return processed_img, results, time_ms