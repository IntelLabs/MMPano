import cv2
import numpy as np, math
from dataclasses import dataclass

from PIL import Image
import torch
import torchvision.transforms as transforms

from typing import List, Union
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse

import scipy.ndimage as ndimage
import os
import glob
import io
import json
from scipy.ndimage import distance_transform_edt


def cv2_to_pil(image):
    # Convert the cv2 image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the cv2 image to a PIL image
    pil_image = Image.fromarray(image)

    return pil_image


def pil_to_cv2(image):
    # Convert the PIL image to a numpy array
    np_image = np.array(image)

    # Convert the numpy array to a cv2 image
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    return cv2_image


def pil_to_tensor(image):
    # Define a transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Apply the transformation to the PIL image
    tensor_image = transform(image)

    # Add the batch dimension
    return tensor_image.unsqueeze(0)


def pil_mask_to_tensor(pil_mask):
    # Define the transformation to convert the PIL image to a tensor
    transform = transforms.ToTensor()

    # Apply the transformation to the PIL image
    tensor_mask = transform(pil_mask)

    # Repeat the tensor along the channel dimension to create 3 channels
    tensor_mask = tensor_mask.repeat(3, 1, 1)

    # Add the batch dimension
    return tensor_mask.unsqueeze(0)


def mask_to_pil(mask):
    # Multiply the mask by 255 to get values between 0 and 255
    mask = mask * 255

    # Convert the mask to an 8-bit integer numpy array
    mask = np.uint8(mask)

    # Create a black and white PIL image from the mask
    pil_image = Image.fromarray(mask, mode="L")

    return pil_image


def cv2_to_tensor(image):
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to range 0.0 to 1.0
    image_normalized = image_rgb.astype(np.float32) / 255.0

    # Transpose the image array to have the shape (3, h, w)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    # Convert the numpy array to a PyTorch tensor and add a batch dimension
    tensor = torch.from_numpy(image_transposed).unsqueeze(0)

    return tensor


def tensor_to_cv2(tensor_image):
    # Convert tensor image to a numpy array
    image_np = (tensor_image * 255).numpy().astype(np.uint8)

    # Transpose the numpy array to have the shape (h, w, 3)
    image_np_transposed = np.transpose(image_np, (0, 2, 3, 1))

    # Remove the batch dimension
    image_cv2_float = np.squeeze(image_np_transposed, axis=0)

    # Convert the RGB order to BGR order
    return cv2.cvtColor(image_cv2_float, cv2.COLOR_RGB2BGR)


def tensor_mask_to_numpy(mask):
    # Convert the mask to a numpy array
    mask_np = mask.numpy()

    # Remove the batch and channel dimensions
    return np.squeeze(mask_np, axis=(0, 3))


def save_mask_to_png(mask, filename):
    assert len(mask.shape) == 3 and mask.shape[-1] == 1, "Invalid mask shape. Expected (h, w, 1)."

    # Convert the mask to an integer array with values in the range [0, 255]
    mask_255 = (mask * 255).astype(np.uint8)

    # Repeat the single channel mask to create a 3 channel image
    mask_3_channels = np.repeat(mask_255, 3, axis=-1)

    # Save the image
    img = Image.fromarray(mask_3_channels)
    img.save(f"{filename}")



def warp_image_v2(image, K_original, K_new, R, output_size):
    """
    Warp an image to a new view given the original and new camera intrinsic matrices, relative rotation,
    and output image size.
    
    Parameters:
        image (numpy.ndarray): The original image.
        K_original (numpy.ndarray): The original camera's intrinsic matrix.
        K_new (numpy.ndarray): The new camera's intrinsic matrix.
        R (numpy.ndarray): The relative rotation matrix.
        output_size (tuple): The desired output image size (width, height).
    
    Returns:
        warped_image (numpy.ndarray): The warped image.
        mask (numpy.ndarray): Mask indicating if a pixel in the warped image has a corresponding pixel in the original image.
    """

    # Compute the transformation matrix using the scaled new camera intrinsic
    T = K_new.dot(R).dot(np.linalg.inv(K_original))

    # Warp the image using the new transformation matrix to the specified output size
    warped_image = cv2.warpPerspective(image, T, output_size)

    # Create and warp the mask
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(mask, T, output_size, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Convert mask to binary (0 or 1)
    mask_binary = (mask_warped > 250).astype(np.uint8)

    return warped_image, mask_binary

# compute the nearest unmasked region
def mask_to_NN_v2(mask, invert = False):
    # print("mask = {}/{}".format(mask, mask.shape))
    # Set a threshold value to create a binary mask
    threshold = 0.5
    binary_mask = mask > threshold
    
    if invert:
        # Invert the binary_mask to find the unmasked pixels
        binary_mask = 1 - binary_mask

    # Convert the inverted_mask to a NumPy array
    inverted_mask_np = binary_mask

    # Compute the distance transform on the inverted mask
    distance_transform = ndimage.distance_transform_edt(inverted_mask_np)

    # Convert the distance transform back to a PyTorch tensor
    return torch.tensor(distance_transform, dtype=torch.float32)


def generate_left_right_fullPano_pattern(max_step = 8, step_size = 42, final_step = 42):
    pattern = []

    start_step = 1
    angle_begin = step_size
    angle_end = (360 - step_size * (max_step // 2 - 1) + step_size * (max_step // 2)) / 2
    step_mid = angle_end - step_size * (max_step // 2)
    for step in range(start_step, max_step+1):
        if step <= max_step // 2:
            pattern.append((0, angle_begin, 0))
            angle_begin += step_size
        else:
            pattern.append((0, angle_end, 0))
            if step != (max_step // 2 + 1):
                angle_end += step_size
            else:
                angle_end += step_mid

    print(f"pattern = {pattern}")
    return pattern


def create_rotation_matrix(x_angle_degrees, y_angle_degrees, z_angle_degrees):
    x_angle_radians = np.radians(x_angle_degrees)
    y_angle_radians = np.radians(y_angle_degrees)
    z_angle_radians = np.radians(z_angle_degrees)

    cos_x, sin_x = np.cos(x_angle_radians), np.sin(x_angle_radians)
    cos_y, sin_y = np.cos(y_angle_radians), np.sin(y_angle_radians)
    cos_z, sin_z = np.cos(z_angle_radians), np.sin(z_angle_radians)

    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])

    R_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])

    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]])

    R = R_y @ R_x @ R_z

    return R

def read_file_into_list(file_path):
    # Initialize an empty list to hold the lines
    lines_list = []

    # Open the file in read mode ('r')
    with io.open(file_path, 'r', encoding='utf8') as file:
        # Read each line in the file
        for line in file:
            # Add the line to the list (removing any trailing whitespace characters)
            lines_list.append(line.rstrip())
    
    # Return the list of lines
    return lines_list

def save_dict_to_file(dict_obj, file_name):
    with open(file_name, 'w') as file:
        json.dump(dict_obj, file)


def load_dict_from_file(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)



def check_fov_overlap_simplified(rotation_matrix, fov1):
    """
    Simplified check if there is an overlap in the field of view of two images based on rotation angle.

    Parameters:
    rotation_matrix (numpy.ndarray): 3x3 rotation matrix from image1 to image2
    fov1 (tuple): Field of view of image1 (horizontal_angle, vertical_angle) in degrees

    Returns:
    bool: True if there is an overlap, False otherwise
    """

    # Calculate the rotation angle from the rotation matrix
    rotation_angle_rad = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    rotation_angle_deg = np.degrees(rotation_angle_rad)

    # # Compare with the FOV (considering the larger of the horizontal or vertical FOV)
    return rotation_angle_deg <= fov1




@dataclass
class vp90Codec:
    interval_deg: float = 1.0
    fps: float = 60
    video_codec = "VP09"
    video_format = ".webm"


@dataclass
class mp4vCodec:
    interval_deg: float = 0.5
    fps: float = 60
    video_codec = "mp4v"
    video_format = ".mp4"


@dataclass
class mp4Codec:
    interval_deg: float = 0.5
    fps: float = 60
    video_codec = "h264"
    video_format = ".mp4"
