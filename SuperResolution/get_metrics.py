"""image_quality_metrics.py

Contains functions (PSNR, SSIM, LPIPS) to calculate processed images.

Author(s): Isha Ruparelia, Prithvi Seran
"""

# external
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_psnr( original_images: np.ndarray, compressed_images: np.ndarray, max_pixels: int) -> np.ndarray:
    """Return the ratio of the maximum value of the pixel to noise (MSE) which affects the quality of the pixels for a batch of original and compressed images.

    Parameters:
    - original_images: NumPy array, batch of original image metrics.
    - compressed_images: NumPy array, batch of compressed image metrics.
    - max_pixels: integer, maximum pixel values of the images.

    Return:
        - psnr: Array of PSNR values for each image in the batch.
    """
    mse = np.mean(np.square(original_images - compressed_images), axis=(1, 2, 3))
    psnr = 10 * np.log10((max_pixels**2) / mse)
    return psnr


def calculate_ssim( original_images: np.ndarray, compressed_images: np.ndarray, C1: float = 0.01, C2: float = 0.03) -> np.ndarray:
    """Return the SSIM values for a batch of both original and compressed images.

    Parameters:
        - original_images: NumPy array, batch of original image metrics.
        - compressed_images: NumPy array, batch of compressed image metrics.
        - C1: constant value for stability in the formula.
        - C2: constant value for stability in the formula.

    Return:
        - ssim: Array of SSIM values for each image in the batch.
    """
    mu_x = np.mean(original_images, axis=(1, 2, 3))
    mu_y = np.mean(compressed_images, axis=(1, 2, 3))
    sigma_x = np.var(original_images, axis=(1, 2, 3))
    sigma_y = np.var(compressed_images, axis=(1, 2, 3))
    sigma_xy = np.mean(
        (original_images - mu_x[:, np.newaxis, np.newaxis, np.newaxis])
        * (compressed_images - mu_y[:, np.newaxis, np.newaxis, np.newaxis]),
        axis=(1, 2, 3),
    )

    # sigma_xy holds the covariance between the pixel clarity of both the
    # original and compressed images.

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim


def preprocess_image_batch(images: np.ndarray) -> torch.Tensor:
    """Preprocess a batch of images."""
    preprocess = transforms.Compose(
        [
            # convert imported image into PyTorch tensor.
            transforms.ToTensor(),
            # normalize tensor by subtracting the mean values
            # and dividing by the standard deviations.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # add extra dimension at the start of the tensor and return preprocessed
    # image as PyTorch sensor.
    for image in images:
        preprocessed_images = torch.stack([preprocess(image)])
    return preprocessed_images


def calculate_lpips(
    original_images: np.ndarray, compressed_images: np.ndarray, perceptual_model
) -> np.ndarray:
    """Return the LPIPS distances for a batch of original and compressed images.

    Parameters:
    - original_images: NumPy array, batch of original image metrics.
    - compressed_images: NumPy array, batch of compressed image metrics.
    - perceptual_model: Pre-trained model for perceptual similarity.

    Returns:
    - distances: Array of LPIPS distances for each pair of images in the batch.
    """
    preprocessed_original = preprocess_image_batch(original_images)
    preprocessed_compressed = preprocess_image_batch(compressed_images)

    features_original = perceptual_model(preprocessed_original)
    features_compressed = perceptual_model(preprocessed_compressed)

    distances = torch.nn.functional.pairwise_distance(
        features_original, features_compressed
    )
    return distances

if __name__ == "__main__":


    #original images 
    indian_pines = np.load("../SuperResolution/dcscn-super-resolution/indian_pine_array.npy")

    cuprite = np.array(Image.open("../SuperResolution/dcscn-super-resolution/cuprite.png"))

    cuprite = cuprite[:, :, 0:3]

    cuprite = cv2.resize(cuprite, (680, 672))

    ksc = np.array(Image.open("../SuperResolution/dcscn-super-resolution/ksc.png"))

    ksc = ksc[:, :, 0:3]

    ksc = cv2.resize(ksc, (544, 532))

    pavia = np.array(Image.open("../SuperResolution/dcscn-super-resolution/pavia.png"))

    pavia = pavia[:, :, 0:3]
    pavia = cv2.resize(pavia, (536, 536))

    #Super resolution corection images
    indian_pines_corrected = np.array(Image.open("../SuperResolution/dcscn-super-resolution/output/dcscn_L12_F196to48_Sc4_NIN_A64_PS_R1F32/image1DownScaled_result.png"))

    cuprite_corrected = np.array(Image.open("../SuperResolution/dcscn-super-resolution/output/dcscn_L12_F196to48_Sc4_NIN_A64_PS_R1F32/cuprite-smaller_result.png"))

    ksc_corrected = np.array(Image.open("../SuperResolution/dcscn-super-resolution/output/dcscn_L12_F196to48_Sc4_NIN_A64_PS_R1F32/ksc-smaller_result.png"))

    pavia_corrected = np.array(Image.open("../SuperResolution/dcscn-super-resolution/output/dcscn_L12_F196to48_Sc4_NIN_A64_PS_R1F32/pavia-smaller_result.png"))

    #input images

    pavia_input = np.array(Image.open("../SuperResolution/pavia-smaller.png"))

    print("Original Images: \n")

    print(indian_pines.shape)
    print(cuprite.shape)
    print(ksc.shape)
    print(pavia.shape)

    print("Corrected Images: \n")

    print(indian_pines_corrected.shape)
    print(cuprite_corrected.shape)
    print(ksc_corrected.shape)
    print(pavia_corrected.shape)

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Display first image on the first subplot
    axes[0].imshow(pavia)
    axes[0].set_title('Image 1')

    # Display second image on the second subplot
    axes[1].imshow(pavia_corrected)
    axes[1].set_title('Image 2')

    # Display second image on the second subplot
    axes[2].imshow(pavia_input)
    axes[2].set_title('Image 3')

    # Remove axes for better clarity
    for ax in axes:
        ax.axis('off')

    original_images = [cuprite, ksc, pavia]

    corrected_images = [cuprite_corrected, ksc_corrected, pavia_corrected]

    total_psnr = 0
    total_ssim = 0

    for i in range(len(original_images)):

        pnsr = calculate_psnr(np.array([original_images[i]]), np.array([corrected_images[i]]), np.max(original_images[i]))

        ssim = calculate_ssim(np.array([original_images[i]]), np.array([corrected_images[i]]), np.max(original_images[i]))

        total_psnr = total_psnr + pnsr
        total_ssim = total_ssim + ssim

    avg_psnr = total_psnr/len(original_images)
    avg_ssim = total_ssim/len(original_images)

    print("Avg PSNR of corrected images compared with original images: \n")
    print(avg_psnr)
    print("Avg SSIM of corrected images compared with original images: \n")
    print(avg_ssim)

    print("Images tested on are Cuprite, Pavia, and KSC")





    




    

