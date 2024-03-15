#! /usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
from typing import Dict, Optional, Tuple

from PIL import Image
import torch
from scipy.ndimage import distance_transform_edt

import json
import re
import shutil
import argparse
from pprint import pprint
from tqdm import tqdm

import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
from utils.common import (
    Descriptor,
    extract_words_after_we_see_withFailv3,
    extract_words_after_we_see_withFailv2
)
from utils.image_utils import (
    cv2_to_pil, pil_to_cv2, mask_to_pil,
    vp90Codec, mp4vCodec, mp4Codec, warp_image_v2, 
    mask_to_NN_v2, generate_left_right_fullPano_pattern,
    create_rotation_matrix, read_file_into_list, save_dict_to_file,
    load_dict_from_file, check_fov_overlap_simplified,
)
from utils.model_utils import (
    is_on_hpu, load_diffusion_model,
    load_blip_model_and_processor, load_upscaler_model,
)
from utils.llm_engines import get_llm_engine, _VALIDATED_MODELS

import math

from typing import List, Union

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


def extract_valid_lines(text, start_with: str = None):
    lines = text.split('\n')
    valid_lines = [line for line in lines if line.strip()]
    if start_with:
        valid_lines = [line for line in lines if line.startswith(start_with) and line.strip()]
    else:
        valid_lines = [line for line in lines if line.strip()]
    return valid_lines


def compute_merge_weight(dis_to_mask):
    return dis_to_mask


# compute the zero-padding range in 4 directions of the image plane so that the field of view is at least a certain number
def compute_padding_range(intrinsics, w, h, fov_min_half = np.pi / 4):
    # compute the ratio required by fov_min_half
    ratio_min = np.tan(fov_min_half)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    p_left = max(0, fx * (ratio_min) - cx)
    p_right = max(0, fx * (ratio_min) - (w-cx))
    p_down = max(0, fy * (ratio_min) - cy)
    p_up = max(0, fy * (ratio_min) - (h-cy))
    return p_left, p_right, p_down, p_up


def find_kth_minmax(np_array, k = 10):
	# np_array (1, N, M), find for each M dim
	out_diff = []
	output_min = []
	for i in range(np_array.shape[-1]):
		out_diff.append((-np.partition(-np_array[0,:,i], k)[k])-(np.partition(np_array[0,:,i], k)[k]))
	return out_diff

def is_uniform_color(pil_image, tolerance=10):
	# Convert the PIL Image to a NumPy array
	image = np.array(pil_image)

	# Calculate the height and width of the image
	height, width, _ = image.shape
	# Check each row on the top and bottom
	for i in range(10):
		if i ==0:
			for row in [image[i:i+1], image[-(i+1):]]:
				out_diff = find_kth_minmax(row.astype(np.float32))
				if out_diff[0] <= tolerance and out_diff[1] <= tolerance and out_diff[2] <= tolerance:
					return True
				
			# Check each column on the left and right
			for column in [image[:,i:i+1], image[:,-i-1:]]:
				column = column.transpose(1, 0, 2)
				out_diff = find_kth_minmax(column.astype(np.float32))
				if out_diff[0] <= tolerance and out_diff[1] <= tolerance and out_diff[2] <= tolerance:
					return True
		else:
			for row in [image[i:i+1], image[-(i+1):-i]]:
				out_diff = find_kth_minmax(row.astype(np.float32))
				if out_diff[0] <= tolerance and out_diff[1] <= tolerance and out_diff[2] <= tolerance:
					return True
                
			# Check each column on the left and right
			for column in [image[:,i:i+1], image[:,-i-1:-i]]:
				column = column.transpose(1, 0, 2)
				out_diff = find_kth_minmax(column.astype(np.float32))
				if out_diff[0] <= tolerance and out_diff[1] <= tolerance and out_diff[2] <= tolerance:
					return True
				
	return False


def create_panorama(image, intrinsic, output_folder, processor, img2text_pipe, inpaint_pipe, sr_pipe, device,
                    sr_inf_step = 75, cinpaint_th = 32., init_prompt = None, major_obj_number = 2,
                    torch_dtype: torch.dtype = torch.float16, 
                    panorama_descriptor: Optional[Dict] = None, llm_engine = None):

    height, width, _ = image.shape
    height_resize, width_resize = 512, 512

    image_pil = cv2_to_pil(cv2.resize(image, (height_resize, width_resize), interpolation=cv2.INTER_LINEAR))
    
    if init_prompt in [None, ""]:
        prompt = "Question: What is this place (describe with fewer than 5 words)? Answer:"
        inputs = processor(image_pil, text=prompt, return_tensors="pt").to(device, torch_dtype)
        generated_ids = img2text_pipe.generate(**inputs, max_new_tokens=15)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        init_prompt = generated_text
    else:
        init_prompt = init_prompt

    prompt_distance = 'No close objects or walls. '

    orig_prompt_init = init_prompt + '. ' + prompt_distance + "Ultra realistic, epic, exciting, wow, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k"
    orig_negative_prompt = "multiple subfigures, close objects, large objects, human, people, pedestrian, close view, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

    print("expanding the fov for the input image...")

    os.makedirs(output_folder, exist_ok = True)
    
    # save the BLIP description
    intrinsics = np.array([[intrinsic[0]*width, 0., intrinsic[2]*width], 
                           [0., intrinsic[1]*height, intrinsic[3]*height],
                           [0., 0., 1.]]).astype(np.float32)
    p_left, p_right, p_down, p_up = compute_padding_range(intrinsics, width, height, np.arctan(1.19)) # half field of view = ~50 deg (to create sufficient overlaps for top/bot view)     

    if max(p_left, p_right, p_down, p_up) <= 0:
        image = cv2.resize(image, (height_resize, width_resize), interpolation=cv2.INTER_LINEAR)
        # TODO(Joey Chou): Ask if this is needed
        scale_x, scale_y = float(width_resize) / width, float(height_resize) / height
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
    else:
        # pad images and create mask
        # compute the 4 corners of the original image, and fit it into the resized image range
        # take the max to make the resized image square
        wh_new = int(max(p_left + p_right + width, p_down + p_up + height)) 

        # left right
        width_ori_resize, height_ori_resize = math.ceil(width_resize / wh_new * width), math.ceil(height_resize / wh_new * height)
        # corner_location
        loc_corner = (width_resize//2 - width_ori_resize//2, height_resize // 2 - height_ori_resize//2)
        # create the new image and put the resized original image into it, and create the mask and new intrinsics
        image_resized = np.zeros((512, 512, 3), dtype = image.dtype)
        image_resized[loc_corner[1]:loc_corner[1]+height_ori_resize, loc_corner[0]:loc_corner[0]+width_ori_resize] = cv2.resize(image, (width_ori_resize, height_ori_resize), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = float(width_ori_resize) / width, float(height_ori_resize) / height
        image = image_resized
        # create mask
        mask = np.ones((height_resize, width_resize), dtype = np.float32)
        mask[loc_corner[1]:loc_corner[1]+height_ori_resize, loc_corner[0]:loc_corner[0]+width_ori_resize] = 0.0
        # save the resized first image
        cv2.imwrite(output_folder + '/input_before_inpaint.png', image)

        # inpaint the masked region
        mask_revert = mask
        pil_image = cv2_to_pil(image)    
        
        # detect whether there is a pure-colored top/down region
        pure_color_bg = True
        while pure_color_bg:
            image_inpaint = inpaint_pipe(prompt=orig_prompt_init, negative_prompt=orig_negative_prompt, image=pil_image, mask_image=mask_revert, num_inference_steps=25).images[0]
            pure_color_bg = is_uniform_color(image_inpaint)
            # # for test
            if pure_color_bg:
                image_inpaint.save(output_folder + '/test_pure_color.png')
            print("do we have pure color for the inpainted image? {}".format(pure_color_bg))

        # make inpainting consistent
        mask_cinpaint = (1-mask)
        dist2zero = distance_transform_edt(mask_cinpaint)
        # 2. build weight map according to dist2zero
        weight_map_cinpaint  = np.ones(mask_cinpaint.shape).astype(np.float32)
        weight_map_cinpaint[dist2zero<=cinpaint_th] = dist2zero[dist2zero<=cinpaint_th]/cinpaint_th        
        image_inpaint = Image.fromarray((np.array(pil_image) * weight_map_cinpaint[:,:,np.newaxis] + np.array(image_inpaint) * (1-weight_map_cinpaint)[:,:, np.newaxis]).astype('uint8'))

        # perform super-resolution on the inpainted image
        # dont use the SR result for warp-and-inpaint
        # save it as another mesh so that we merge it into the inpainted image later
        if sr_pipe is not None:
            upscaled_image = sr_pipe(prompt=orig_prompt_init, negative_prompt=orig_negative_prompt, image=image_inpaint, num_inference_steps = sr_inf_step).images[0]

        prompt = "Question: Describe the foreground and background in detail and separately? Answer:"
        inputs = processor(pil_image, text=prompt, return_tensors="pt").to(device, torch_dtype)
        generated_ids = img2text_pipe.generate(**inputs, max_new_tokens=15)
        generated_text_details = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        image = pil_to_cv2(image_inpaint)

        # create intrinsics
        intrinsics[0,2] = loc_corner[0] + intrinsics[0,2] * scale_x
        intrinsics[1,2] = loc_corner[1] + intrinsics[1,2] * scale_y
        intrinsics[0,0] *= scale_x
        intrinsics[1,1] *= scale_y

        if sr_pipe is not None:
            image_SR = pil_to_cv2(upscaled_image)
            # intrinsics for image_SR
            intrinsics_SR = np.copy(intrinsics)
            intrinsics_SR[0,:] *= 4
            intrinsics_SR[1,:] *= 4

    # save the resized first image
    cv2.imwrite(output_folder + '/input_resized.png', image)
    

    image_list = [image]
    pose_list = [(0, 0, 0)]
    if sr_pipe is not None:
        image_SR_list = [image_SR]
        cv2.imwrite(output_folder + '/input_resized_SR.png', image_SR)

    max_step = 6
    step_size = 41
    vortex_list = generate_left_right_fullPano_pattern(max_step=max_step, step_size = step_size, final_step = 55)
    
    if not panorama_descriptor:
        question_for_llm = "Given a scene with {}, where in font of us we see {}. MUST generate {} rotated views to describe what else you see in this place, where the camera of each view rotates {} degrees to the right (you dont need to describe the original view, i.e., the first view of the {} views you need to describe is the view with {} degree rotation angle). Dont involve redundant details, just describe the content of each view. Also don't repeat the same object in different views. Don't refer to previously generated views. Generate concise (< 10 words) and diverse contents for each view. Each sentence starts with: View xxx(view number, from 1-{}): We see.... (don't mention any information about human, animal or live creature)".format(init_prompt, generated_text_details, max_step, 360//max_step, max_step, 360//max_step, max_step)
        question_for_llm_major_object = "Given a scene with {}, where in font of us we see {}. What would be the two major foreground objects that we see? use two lines to describe them where each line is in the format of 'We see: xxx (one object, dont describe details, just one word for the object. Start from the most possible object. Don't mention background objects like things on the wall, ceiling or floor.)'".format(init_prompt, generated_text_details)
        question_for_llm_remove_objects = "Modify the sentence: '{}' so that we remove all the objects from the description (e.g., 'a bedroom with a bed' would become 'a bedroom'. Do not change the sentence if the description is only an object). Just output the modified sentence.".format(init_prompt)

        # We want to repeat this process until there is no human detected in the answer
        _message, history = llm_engine.chat(question_for_llm)
        question_remove_animal = 'given the description of multiple views: \'{}\' remove any information about human, animal, or live creature in the descriptions. Answer with simply the modified content, i.e., View XXX (view number): We see ... (contents without human info)'.format(_message) 
        message, _ = llm_engine.chat(question_remove_animal, history=history)

        message_main_obj, _ = llm_engine.chat(question_for_llm_major_object, history=None)
        description_no_obj, _ = llm_engine.chat(question_for_llm_remove_objects, history=None)

        lines = extract_valid_lines(message, start_with="View")
        lines_major_obj = extract_valid_lines(message_main_obj)

        while len(lines_major_obj) != 2 or extract_words_after_we_see_withFailv2(lines_major_obj[0]) is None or extract_words_after_we_see_withFailv2(lines_major_obj[1]) is None:
            message_main_obj, _ = llm_engine.chat(question_for_llm_major_object, history=None)
            lines_major_obj = extract_valid_lines(message_main_obj)
        
        if len(lines) != (max_step):
            print("[error] num_lines != {}".format(max_step))
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(len(lines), max_step)
            print(message)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            return 1

        is_repeated = []
        is_repeated_all = True
        num_false = 0
        for obj_id in range(major_obj_number):
            # let LLM decide whether the object is repeating
            question_for_llm_repeat = "Do we often see multiple {} in a scene with {}? Just say 'yes' or 'no' with all lower case letters".format(extract_words_after_we_see_withFailv2(lines_major_obj[obj_id]), init_prompt)
            fail = True
            while fail:
                message_repeat, _ = llm_engine.chat(question_for_llm_repeat)
                
                if 'yes' or 'Yes' in message_repeat:
                    is_repeated.append(True)
                    fail = False
                elif 'no' or 'No' in message_repeat:
                    is_repeated.append(False)
                    fail = False
                    is_repeated_all = False
                    num_false += 1
                else:
                    print(f"wrong output for repeat answer = {message_repeat}")
         
        # Create dictionary
        panorama_descriptor = Descriptor(**{
            "init_prompt": init_prompt,
            "generated_text_details": generated_text_details,
            "message": message,
            "message_main_obj": message_main_obj,
            "question_for_llm_repeat": question_for_llm_repeat,
            "description_no_obj": description_no_obj,
            "major_obj_number": major_obj_number,
            "is_repeated": is_repeated,
        })
    else:
        init_prompt = panorama_descriptor.init_prompt
        generated_text_details = panorama_descriptor.generated_text_details
        message = panorama_descriptor.message
        message_main_obj = panorama_descriptor.message_main_obj
        question_for_llm_repeat = panorama_descriptor.question_for_llm_repeat
        description_no_obj = panorama_descriptor.description_no_obj
        major_obj_number = panorama_descriptor.major_obj_number
        is_repeated = panorama_descriptor.is_repeated

        lines = extract_valid_lines(message)
        lines_major_obj = extract_valid_lines(message_main_obj)
        is_repeated_all = all(is_repeated)

    panorama_descriptor.save_json(os.path.join(output_folder, "panorama_descriptor.json"))

    print("====================================================================")
    print("LLM descriptions:")
    pprint(panorama_descriptor)
    print("====================================================================")

    order = [5, 0, 1, 4, 2, 3]
    for i in order:
        pose = vortex_list[i]
        print("generating view i = {}, pose = {}".format(i, pose))
        # generate the warped image and mask
        rotation_matrix = create_rotation_matrix(pose[0], pose[1], pose[2])

        # Warp previous images to the new view and create the mask
        warped_image_accumulate = np.zeros((height_resize, width_resize, 3), dtype = np.float32)
        weight_accumulate = np.zeros((height_resize, width_resize, 1), dtype = np.float32)
        mask_accumulate = np.zeros((height_resize, width_resize), dtype = np.float32)
        
        if sr_pipe is not None:
            warped_image_accumulate_SR = np.zeros((height_resize, width_resize, 3), dtype = np.float32)
        
        for j in range(len(image_list)):
            # get the relative pose from the j-th image to the current view
            pose_prev = pose_list[j]
            rotation_matrix_prev = create_rotation_matrix(pose_prev[0], pose_prev[1], pose_prev[2])
            rot_mat_prev_to_curr = rotation_matrix_prev.T @ rotation_matrix

            # skip non-overlapping views
            is_overlap = check_fov_overlap_simplified(rot_mat_prev_to_curr, 80) # hard-coded small fovs so that we only keep neighbouring images
            if not is_overlap:
                print("pose: {} is not overlapping with the current image ({}/{})".format(pose_prev, i, pose))
                continue

            # warp image
            warped_image, mask = warp_image_v2(image_list[j], intrinsics, intrinsics, rot_mat_prev_to_curr.T, (height_resize, width_resize))
            dis_to_mask = mask_to_NN_v2(mask) # [h, w]
            # render, disp, mask, dis_to_mask, rads = rgbd_renderer.render_mesh_with_normal(mesh_list[j], intrinsics_tensor, ext_tensor)
            # render SR and prepare a hgih_res warped image for later use
            if sr_pipe is not None:
                warped_image_SR, mask_SR = warp_image_v2(image_SR_list[j], intrinsics_SR, intrinsics_SR, rot_mat_prev_to_curr.T, (height_resize*4, width_resize*4))

            if sr_pipe is not None:
                warped_image_SR = cv2.resize(warped_image_SR, (width_resize, height_resize), interpolation = cv2.INTER_CUBIC)

            weight_map = compute_merge_weight(dis_to_mask).numpy()
            # accumulate the warped image with weights
            warped_image_accumulate += warped_image.astype(np.float32) * weight_map[:, :, np.newaxis]
            if sr_pipe is not None:
                warped_image_accumulate_SR += warped_image_SR.astype(np.float32) * weight_map[:, :, np.newaxis]
            weight_accumulate += weight_map[:,:, np.newaxis]
            mask_accumulate[mask == 1] = 1.0

        zero_indices = (weight_accumulate == 0)
        weight_accumulate[zero_indices] = 1.0
        warped_image = np.clip((warped_image_accumulate/weight_accumulate).astype(np.uint8), 0, 255)
        mask = mask_accumulate
        if sr_pipe is not None:
            warped_image_SR = np.clip((warped_image_accumulate_SR/weight_accumulate).astype(np.uint8), 0, 255)

        # set the line number to get the right line:
        description = extract_words_after_we_see_withFailv3(lines[i])
        if description is None:
            print("[error] GPT prompt not following our format: {}".format(lines[i]))
            return 1
        else:
            # get the answer for repeated objects
            if is_repeated_all:
                print("the major objects are repeated")
                description = 'a peripheral view of {} where we see{} {}'.format(description_no_obj, description, prompt_distance)
            else:
                description = 'a peripheral view of {} where we only see{} {}'.format(description_no_obj, description, prompt_distance)

        orig_prompt = description+", ultra realistic, epic, exciting, wow, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k, large field of view (100 degrees)"
        if not is_repeated_all:
            orig_negative_prompt = "human, people, pedestrian, close objects, large objects, close view, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"
            for obj_id in range(major_obj_number):
                orig_negative_prompt = 'any type of ' + extract_words_after_we_see_withFailv2(lines_major_obj[obj_id]) + ', ' + orig_negative_prompt
        else:
            orig_negative_prompt = "human, people, pedestrian, close objects, large objects, close view, mirror, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

        mask_revert = mask_to_pil(1-mask)
        if sr_pipe is not None:
            pil_image = cv2_to_pil(warped_image_SR)
        else:
            pil_image = cv2_to_pil(warped_image)
        # detect whether there is a pure-colored top/down region
        pure_color_bg = True
        iter_count = 0
        while pure_color_bg and iter_count < 20:
            image_inpaint = inpaint_pipe(prompt=orig_prompt, negative_prompt=orig_negative_prompt, image=pil_image, mask_image=mask_revert, num_inference_steps=25).images[0]
            pure_color_bg = is_uniform_color(image_inpaint)
            print("[avoid pure color background] do we have pure color for the inpainted image? {}".format(pure_color_bg))

            for obj_id in range(major_obj_number):
                if not is_repeated_all and not is_repeated[obj_id] and pose[0] == 0 and i > 0 and i < 5:
                    prompt = "Question: is there a {} in this picture (just say yes or no)? Answer:".format(extract_words_after_we_see_withFailv2(lines_major_obj[obj_id]))
                    inputs = processor(image_inpaint, text=prompt, return_tensors="pt").to(device, torch_dtype)
                    generated_ids = img2text_pipe.generate(**inputs, max_new_tokens=15)
                    generated_text_repeat = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    print("repeated check = {}".format(generated_text_repeat))
                    if "yes" in generated_text_repeat:
                        print(" we see {} in the inpainted view".format(extract_words_after_we_see_withFailv2(lines_major_obj[obj_id])))
                        pure_color_bg = True
                        iter_count += (1.0/num_false)
                        if not is_repeated_all and iter_count >= 20:
                            is_repeated_all = True
                            print("reaching maximum checking iterations, there is a conflict, setting is_repeated to true")
        inpainted_cv2 = pil_to_cv2(image_inpaint)

        # we do the same merging step as the
        # 1. compute the weight mask for the warped image
        dist2zero = distance_transform_edt(mask_accumulate)

        # 2. build weight map according to dist2zero
        weight_map_cinpaint = np.ones(mask_accumulate.shape).astype(np.float32)
        weight_map_cinpaint[dist2zero <= cinpaint_th] = dist2zero[dist2zero <= cinpaint_th] / cinpaint_th

        # Save image at each step
        if sr_pipe is not None:
            inpainted_cv2_merge = warped_image_SR * weight_map_cinpaint[:, :, np.newaxis] + inpainted_cv2 * (1 - weight_map_cinpaint)[:, :, np.newaxis]
            # filename = os.path.join(output_folder, f"inpaint_step_SR_{i}.png")
        else:
            inpainted_cv2_merge = warped_image * weight_map_cinpaint[:, :, np.newaxis] + inpainted_cv2 * (1 - weight_map_cinpaint)[:, :, np.newaxis]
            # filename = os.path.join(output_folder, f"inpaint_step_{i}.png")
        filename = os.path.join(output_folder, f"inpaint_step_{i}.png")
        cv2.imwrite(filename, inpainted_cv2_merge)

        # Perform super-resolution on the inpainted_cv2 (not on inpainted_cv2_SR to prevent noise amplification)
        if sr_pipe is not None:
            # image_inpaint_SR = cv2_to_pil(inpainted_cv2.astype(np.uint8))
            image_inpaint_SR = cv2_to_pil(inpainted_cv2_merge.astype(np.uint8))
            image_inpaint_SR = sr_pipe(prompt=orig_prompt, negative_prompt=orig_negative_prompt, image=image_inpaint_SR, num_inference_steps=sr_inf_step).images[0]
            image_inpaint_SR_cv2 = pil_to_cv2(image_inpaint_SR)
            filename = os.path.join(output_folder, f"inpaint_step_SR_{i}.png")
            cv2.imwrite(filename, image_inpaint_SR_cv2)

        image_list.append(inpainted_cv2)
        if sr_pipe is not None:
            image_SR_list.append(image_inpaint_SR_cv2)
        pose_list.append(pose)

    return 0


def parse_args():
    def list_of_num(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser(description='Multimodal Panorama Generation')
    parser.add_argument('--device', type=str, default="hpu", choices=["cpu", "cuda", "hpu"], help="Target HW device for Diffusion and BLIP models")
    parser.add_argument('--dtype', type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="Datatype for model inference.")
    parser.add_argument('--init_prompt', type=str, help='Prompt which will be used for text to panorama generation.')
    parser.add_argument('--init_image', type=str, help='Path to a image which will be used for image to panorama generation.')
    parser.add_argument('--output_folder', type=str, default='./exp/output')
    parser.add_argument('--cpu_offload', action="store_true", help="Flag if user want to offload StableDiffusion pipeline to CPU")

    parser.add_argument('--text2pano', action="store_true", help="Flag if user want to do text-to-panorama. Else will do image-to-panorama.")
    parser.add_argument('--llm_model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        choices=_VALIDATED_MODELS, help='Name of LLM model for text generation.')
    parser.add_argument('--api_key', type=str, default="", help="your OpenAI API key")
    parser.add_argument('--intrinsic', type=list_of_num, default=[1.11733848262, 1.11733848262, 0.5, 0.5], help="Intrinsic.")
    parser.add_argument('--panorama_descriptor', type=str, help='Path to a descriptor JSON that will be used for panorama generation.')

    parser.add_argument('--do_upscale', action="store_true", help="Flag if user want to use super resolution to upscale the generated images")
    parser.add_argument('--major_obj_number', type=int, default=2, choices=[1, 2], help='how many major objects we do we want to consider so that they dont repeat?')
    parser.add_argument('--sr_inf_step', type=int, default=35, help='number of inference steps for the super resolution model')

    parser.add_argument('--inpaint_model_name', type=str, default="stabilityai/stable-diffusion-2-inpainting",
                        help="Diffusion model name")
    parser.add_argument('--blip_model_name', type=str, default="Salesforce/blip2-flan-t5-xl",
                        help="BLIP model name")
    parser.add_argument('--upscaler_model_name', type=str, default="stabilityai/stable-diffusion-x4-upscaler",
                        help="Super resolution upscaler model name")

    # Generate panorama and video
    parser.add_argument('--save_pano_img', action="store_true", help="Flag if user want to save the panorama image.")
    parser.add_argument('--gen_video', action="store_true", help="Flag if user want to generate and save a video of panorama view.")
    parser.add_argument('--video_codec', type=str, default="MP4V", choices=["MP4V", "VP09"],
                        help="Video codec used to generate the video")
    args = parser.parse_args()

    # Validate arguments
    if len(args.intrinsic) != 4:
        raise RuntimeError(f"--intrinsic has to be 4 floating point number. Got {args.intrinsic}")

    return args


def gen_multiviews(
    device: str,
    dtype: str = "float32",
    output_folder: str = "./outputs",
    init_prompt: Optional[str] = None,
    init_image: Optional[Union[str, Image.Image]] = None,
    cpu_offload: bool = False,
    # Text generation
    text2pano: bool = False,
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    api_key: str = "",
    panorama_descriptor: Optional[Union[str, Dict[str, str]]] = None,  # None, path to JSON, or a dictionary
    use_predefine_llm_descriptor: bool = False,
    llm_engine = None,
    # Panorama generation
    intrinsic: List[float] = [1.11733848262, 1.11733848262, 0.5, 0.5],
    do_upscale: bool = False,
    major_obj_number: int = 2,
    sr_inf_step: int = 35,
    inpaint_model_name: Optional[str] = "stabilityai/stable-diffusion-2-inpainting",
    blip_model_name: Optional[str] = "Salesforce/blip2-flan-t5-xl",
    upscaler_model_name: Optional[str] = "stabilityai/stable-diffusion-x4-upscaler",
    text2img_model_name: Optional[str] = "stabilityai/stable-diffusion-2-base", 
    # Pre-loaded pipelines, if any
    inpaint_pipe: Optional = None,
    processor: Optional = None,
    img2text_pipe: Optional = None,
    sr_pipe: Optional = None,
    text2img_pipe: Optional = None,
    **kwargs,
    ):

    if is_on_hpu(device) and dtype == "float16":
        # Force dtype to be bfloat16 on HPU
        dtype = "bfloat16"

    print("===========================================================================")
    print(f"Running Multimodal Panorama Generation on {device} in {dtype}.")
    print("===========================================================================")

    ##################
    # Parse descriptor
    ##################
    # If given, get the pre-generated LLM descriptions
    if panorama_descriptor is not None and use_predefine_llm_descriptor:
        if isinstance(panorama_descriptor, dict):
            panorama_descriptor = Descriptor(**panorama_descriptor)
        elif isinstance(panorama_descriptor, str) and os.path.isfile(panorama_descriptor):
            panorama_descriptor = Descriptor.from_json(panorama_descriptor)
        elif not isinstance(panorama_descriptor, Descriptor):
            raise RuntimeError(f"panorama_descriptor should be a JSON file, Dictionary, or Descriptor type.")

        # If only init_prompt is given in the panorama_descriptor, do the text-to-panorama
        if not panorama_descriptor.init_image:
            assert panorama_descriptor.init_prompt, "At least one of [`init_prompt`, `init_image`] must be given"
            text2pano = True

    elif panorama_descriptor is None and use_predefine_llm_descriptor:
        raise RuntimeError(f"`panorama_descriptor` must be provided when setting `use_predefine_llm_descriptor=True`")

    ######################
    # Create output folder
    ######################
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok = True)
    print(f"Save all outputs to {output_folder}")

    #############################
    # Load pipelines if not given 
    #############################
    # Inpainting pipeline
    if inpaint_pipe is None:
        inpaint_pipe = load_diffusion_model(inpaint_model_name, device=device, dtype=dtype, cpu_offload=cpu_offload)

    # Image-to-text pipeline
    if processor is None and img2text_pipe is None:
        processor, img2text_pipe = load_blip_model_and_processor(blip_model_name, device=device, dtype=dtype)
    elif (processor is not None and img2text_pipe is None) or (processor is None and img2text_pipe is not None):
        raise RuntimeError(
            "Processor and BLIP model has to be set or not set at the same time. "
            f"Got processor={processor}, img2text_pipe={img2text_pipe}."
        )

    # Super resolution
    if sr_pipe is None and do_upscale:
        # NOTE: Skip upscaler in light version
        sr_pipe = load_upscaler_model(upscaler_model_name, device, dtype)

    # Text-to-image
    if text2pano and text2img_pipe is None:
        # Load Diffusion pipeline
        text2img_pipe = load_diffusion_model(text2img_model_name, device=device, dtype=dtype, cpu_offload=cpu_offload)

    # Text generation 
    if llm_engine is None:
        llm_engine = get_llm_engine(llm_model_name, device=device, dtype=dtype, openai_key=api_key)

    ###########################
    # Text or Image to Panorama
    ###########################
    init_prompt = init_prompt if panorama_descriptor is None else panorama_descriptor.init_prompt
    init_image = init_image if panorama_descriptor is None else panorama_descriptor.init_image

    t_begin = time.time()
    # Use given init_image or generate an init_image from the init_prompt.
    # This will be used for generating panorama
    if text2pano:
        print(f"Generating init image with prompt={init_prompt} ...")
        init_image = text2img_pipe(init_prompt, num_inference_steps=25).images[0]
        init_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    elif init_image is not None:
        if isinstance(init_image, str):
            # init_image is a path to a file
            print(f"Loading init image from {init_image}")
            init_image = cv2.imread(init_image, cv2.IMREAD_COLOR)
        elif isinstance(init_image, Image.Image):
            init_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
        elif isinstance(init_image, np.ndarray):
            pass
    else:
        # TODO(Joey Chou): Add error message
        raise RuntimeError("Please do text2pano with a given init_prompt, or pass a init_image to do image to pano")

    # check whether the intrinsic matrix exist
    with torch.inference_mode():
        fail = True
        while fail:
            fail = create_panorama(
                init_image, intrinsic, output_folder, processor, img2text_pipe, inpaint_pipe, sr_pipe, device,
                sr_inf_step, init_prompt=init_prompt, major_obj_number=major_obj_number,
                panorama_descriptor=panorama_descriptor, llm_engine=llm_engine
            )
    print(f"Total runtime: {time.time() - t_begin}")


def _gen_pano_outputs(images: List[np.ndarray],
                      out_dir: str,
                      rotation_degrees: List[int],
                      fov: float = 99.9169018, gen_video: bool = False,
                      save_pano_img: bool = True,
                      # Video related
                      video_size: Tuple[int, int] = (512, 512), video_codec: str = "MP4V",
                      new_pano: Optional = None):
    """
    To make video works with gradio, please use the setup as below:
        * interval_deg = 1.0
        * fps: = 60
        * video_codec = "VP09"

    For other application that works with mp4v:
        * interval_deg = 0.5
        * fps = 60
        * video_codec = "MP4V"
    """

    if new_pano is None:
        _output_image_name = "pano.png"

        ee = m_P2E.Perspective(
                images,
                [
                    [fov, rotation_degrees[0], 0], [fov, rotation_degrees[1], 0], [fov, rotation_degrees[2], 0], [fov, rotation_degrees[3], 0],
                    [fov, rotation_degrees[4], 0], [fov, rotation_degrees[5], 0], [fov, rotation_degrees[6], 0]
                ]
            )

        new_pano = ee.GetEquirec(2048, 4096)

        if save_pano_img:
            # Output panorama image
            cv2.imwrite(os.path.join(out_dir, _output_image_name), new_pano.astype(np.uint8)[540:-540])

    if gen_video:
        if video_codec.upper() == "MP4V":
            codec_config = mp4vCodec()
        elif video_codec.upper() == "VP09":
            codec_config = vp90Codec()
        elif video_codec.upper() == "MP4":
            codec_config = mp4Codec()
        else:
            raise RuntimeError(f"Only support codec ['.MP4V', 'VP09']. Got {video_codec}")

        output_video_name = f"video{codec_config.video_format}"
        interval_deg = codec_config.interval_deg

        video_codec = codec_config.video_codec
        fps = codec_config.fps

        fov = 86
        num_frames = int(360 / interval_deg)

        equ = E2P.Equirectangular(new_pano)
        img = equ.GetPerspective(fov, 0, 0, *video_size)  # Specify parameters(FOV, theta, phi, height, width)

        margin = 0
        if margin > 0:
            img = img[margin:-margin]
        size = (img.shape[1], img.shape[0])

        save_video_path = os.path.join(out_dir, output_video_name)
        print("save_video_path = ", save_video_path, "; ", video_codec, ", ", fps, ", ", size, ", video_size = ", video_size)
        out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*video_codec), fps, size)

        for i in tqdm(range(num_frames)):
            # Process image
            deg = i * interval_deg
            img = equ.GetPerspective(fov, deg, 0, *video_size)  # Specify parameters(FOV, theta, phi, height, width)
            if margin > 0:
                img = img[margin:-margin]
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Write to video
            out.write(img)
        out.release()

        # ffmpeg -y -i /root/app/rest_api/api_output/demo/video.mp4v /root/app/rest_api/api_output/demo/video.avc1
    return new_pano


def gen_pano(images: Optional[List[np.ndarray]] = None,
             output_folder: Optional[str] = None,
             do_upscale: bool = False,
             save_pano_img: bool = True,
             gen_video: bool = True,
             video_codec: str = "MP4V",
             pano: Optional = None,
             **kwargs,
             ):
    # suffix = '_SR' if do_upscale else ""
    suffix = "" 
    image_names = ["input_resized" + suffix + ".png"]
    for i in range(6):
        image_names.append("inpaint_step" + suffix + "_{}.png".format(i))

    rotations = [create_rotation_matrix(0, 0, 0).T]
    rotation_degrees = [0]
    max_step = 6
    step_size = 41
    vortex_list = generate_left_right_fullPano_pattern(max_step=max_step, step_size=step_size, final_step=55)
    for i in range(6):
        rotations.append(create_rotation_matrix(vortex_list[i][0], vortex_list[i][1], vortex_list[i][2]).T)
        rotation_degrees.append(vortex_list[i][1])

    LR_images = []
    # read individual images out
    for image_name in tqdm(image_names):
        LR_images.append(cv2.imread(os.path.join(output_folder, image_name)))

    return _gen_pano_outputs(LR_images, output_folder, rotation_degrees, save_pano_img=save_pano_img, gen_video=gen_video, video_codec=video_codec, new_pano=pano)


if __name__ == "__main__":
    args = parse_args()

    # Generate multiview scenes
    gen_multiviews(**args.__dict__)

    # Generate panorama view and optionally generate video
    gen_pano(**args.__dict__)
