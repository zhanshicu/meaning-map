#####


import logging
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import requests
import glob
from io import BytesIO
import re

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):  
    # Setup logging
    log_file = os.path.join(args.output_dir, 'inference_2.log')
    setup_logging(log_file)
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    logging.info(f"Loading model: {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        logging.warning(
            "The auto inferred conversation mode is %s, while `--conv-mode` is %s, using %s",
            conv_mode, args.conv_mode, args.conv_mode
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    processed_files = 0
    file_paths = glob.glob(os.path.join(args.data_dir, "patch_info_*.csv"))
    process_paths = file_paths[33:]
    for file_path in process_paths:
        logging.info(f"Processing file: {file_path}")
        processed_files += 1
        
        # Extract video_id from the filename
        video_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
        
        # Data
        test_dataset = pd.read_csv(file_path)
        
        test_dataset['image_path'] = test_dataset.apply(lambda row: os.path.join(row['scene'], row['filename']), axis=1)
        image_paths = test_dataset['image_path']
        
        test_dataset['likert_label_predicted'] = ""  # record predictions
        
        total_images = len(image_paths)
        logging.info(f"Processing {total_images} images for file: {file_path}")
        
        for idx, image_path in tqdm(enumerate(image_paths), total=total_images, desc="Processing images"):
            abs_image_path = [os.path.join(f"../cuts_patch/{video_id}", image_path)]
            images = load_images(abs_image_path)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Record the output to the corresponding row in the dataframe
            test_dataset.at[idx, 'likert_label_predicted'] = outputs

            if (idx + 1) % 500 == 0 or (idx + 1) == total_images:  # Log every 10 images or at the end
                logging.info(f"Processed {idx + 1}/{total_images} images")
        
        test_dataset = test_dataset.drop(columns=['image_path'])
        
        # Save predictions for each file
        output_file = os.path.join(args.output_dir, f"preds_{os.path.basename(file_path)}")
        test_dataset.to_csv(output_file, index=False)
        logging.info(f"Saved predictions to: {output_file}")

    logging.info(f"Total files processed: {processed_files}")

# Define the arguments directly
args = argparse.Namespace(
    model_path="llava-v1.5-7b-merged_new",
    model_base=None,
    query="""
    Please assess the meaningfulness of the depicted patch using the following scale: 'very low,' 'low,' 'somewhat low,' 'somewhat high,' 'high,' 'very high.' 
    Provide your response by selecting one of these categories. 
    """,
    conv_mode=None,
    sep=",",
    temperature=0,
    top_p=None,
    num_beams=1,
    max_new_tokens=512,
    data_dir="../patch_info_dir",
    output_dir="../predictions_output"
)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Evaluate the model
eval_model(args)
