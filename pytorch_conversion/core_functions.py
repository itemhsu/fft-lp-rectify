"""
Core Functions Extracted from tai-LPRect-fft.ipynb
Original image rectification functions using FFT analysis
"""
pass
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
pass
pass
def rgb2gray(rgb):
    """Convert RGB image to grayscale using standard weights"""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def imgWrapA(orgImg, a):
    """
    Apply perspective correction to image based on angle parameter
    
    Args:
        orgImg: Input image array
        a: Correction angle parameter
    
    Returns:
        Corrected image array
    """
    column = orgImg.shape[1]
    row = orgImg.shape[0]
    
    # Create coordinate meshgrid
    col_coords, row_coords = np.meshgrid(np.arange(column), np.arange(row))
    
    # Apply perspective transformation
    # This is a simplified version - the original implementation may be more complex
    col_coords_new = col_coords + a * (row_coords - row/2)
    row_coords_new = row_coords
    
    # Ensure coordinates are within bounds
    col_coords_new = np.clip(col_coords_new, 0, column-1)
    row_coords_new = np.clip(row_coords_new, 0, row-1)
    
    # Apply transformation using cv2.remap
    map_x = col_coords_new.astype(np.float32)
    map_y = row_coords_new.astype(np.float32)
    
    corrected_img = cv2.remap(orgImg, map_x, map_y, cv2.INTER_LINEAR)
    
    return corrected_img


def estCorrect(orgImg0, cutoffF=0.8, margin=0.1):
    """
    Main rectification function using FFT analysis
    
    Args:
        orgImg0: Input image array
        cutoffF: Frequency cutoff parameter (default: 0.8)
        margin: Margin parameter (default: 0.1)
    
    Returns:
        Perspective corrected image
    """
    # Apply denoising
    orgImg = cv2.fastNlMeansDenoisingColored(orgImg0, None, 9, 9, 7, 21)
    
    w = orgImg.shape[1]
    h = orgImg.shape[0]
    print(f"Image dimensions: w={w}, h={h}")
    
    # Convert to grayscale
    img = rgb2gray(orgImg)
    
    # Apply vertical difference (edge detection)
    img_diff = np.diff(img, axis=0)
    img_padded = np.pad(img_diff, ((0, 1), (0, 0)), mode='constant')
    
    # Compute 2D FFT
    f = np.fft.fft2(img_padded)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Apply logarithmic scaling
    magnitude_spectrum_log = 20 * np.log(magnitude_spectrum + 1e-8)
    
    # Convert to polar coordinates
    center_y, center_x = np.array(magnitude_spectrum_log.shape) // 2
    y_coords, x_coords = np.ogrid[:magnitude_spectrum_log.shape[0], :magnitude_spectrum_log.shape[1]]
    
    # Calculate polar coordinates
    x_centered = x_coords - center_x
    y_centered = y_coords - center_y
    
    rho = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Convert theta to degrees and normalize to 0-180 range
    theta_deg = (theta * 180 / np.pi + 180) % 180
    
    # Create polar image with 100 angle bins
    polar_bins = 100
    polar_img = np.zeros((int(np.max(rho)) + 1, polar_bins))
    
    for i in range(magnitude_spectrum_log.shape[0]):
        for j in range(magnitude_spectrum_log.shape[1]):
            r = int(rho[i, j])
            angle_bin = int(theta_deg[i, j] * polar_bins / 180)
            if r < polar_img.shape[0] and angle_bin < polar_bins:
                polar_img[r, angle_bin] = max(polar_img[r, angle_bin], magnitude_spectrum_log[i, j])
    
    # Sum over radial direction to get angle profile
    polar_sum = np.sum(polar_img, axis=0)
    
    # Apply gain correction and find maximum
    GAIN = 1.0
    gainStdev = 0.1
    polar_sum[45:56] = (polar_sum[45:56] * GAIN * gainStdev + 
                       polar_sum[45:56] * (1 - gainStdev))
    
    maxIndex = np.argmax(polar_sum)
    print(f"maxIndex = {maxIndex}")
    
    # Calculate offset angle
    offsetDegree = (maxIndex - 50) / 100 * np.pi
    print(f"offsetDegree = {offsetDegree * 180 / np.pi}")
    
    # Apply rectangular correction
    rec_offsetDegree = math.atan(math.tan(offsetDegree) * h / w)
    offsetDegree = rec_offsetDegree
    print(f"rec_offsetDegree = {rec_offsetDegree * 180 / np.pi}")
    
    # Calculate correction parameter
    aEst = np.sin(offsetDegree)
    
    # Apply correction
    full_pix_color0 = np.array(orgImg0)
    correctImg = imgWrapA(full_pix_color0, aEst)
    
    return correctImg


def estCorrect2D(orgImg, cutoffF=0.8, margin=0.1):
    """
    Apply 2D rectification (both horizontal and vertical)
    
    Args:
        orgImg: Input image array
        cutoffF: Frequency cutoff parameter
        margin: Margin parameter
    
    Returns:
        Tuple of (horizontal_corrected, fully_corrected) images
    """
    # Horizontal correction
    hCorrectedImg = estCorrect(orgImg, cutoffF, margin)
    
    # Rotate 90 degrees for vertical correction
    hCorrectedImg90 = np.rot90(hCorrectedImg, k=1)
    vCorrectedImg = estCorrect(hCorrectedImg90, cutoffF, margin)
    
    # Rotate back to original orientation
    vCorrectedImg270 = np.rot90(vCorrectedImg, k=3)
    
    return hCorrectedImg, vCorrectedImg270


def fftplotWarp(imgPath):
    """
    Complete processing pipeline from image file
    
    Args:
        imgPath: Path to input image file
    
    Returns:
        Tuple of (horizontal_corrected, fully_corrected) images
    """
    orgImg = cv2.imread(imgPath)
    pix_color = np.array(orgImg)
    hCorrectedImg, CorrectedImg = estCorrect2D(pix_color, 0.8, 0.1)
    
    return hCorrectedImg, CorrectedImg


# Configuration parameters
DEFAULT_CUTOFF_F = 0.8
DEFAULT_MARGIN = 0.1
DEFAULT_CORE_CUT = 0.01
ENABLE_PLOT = True
ENABLE_BIAS = False
