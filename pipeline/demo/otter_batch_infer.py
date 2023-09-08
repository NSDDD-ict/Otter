import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys
import json
from tqdm import tqdm

sys.path.append("/data/chengshuang/Otter")
from otter.modeling_otter import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Main Function -------------------
load_bit = "fp32"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}
elif load_bit == "fp32":
    precision = {"torch_dtype": torch.float32}

<<<<<<< HEAD
# This model version is trained on MIMIC-IT DC dataset.
model = OtterForConditionalGeneration.from_pretrained("/lustre/S/zhangyang/chengshuang/LLM/Otter/OTTER-Video-LLaMA7B-DenseCaption", device_map="auto", **precision)
# model = OtterForConditionalGeneration.from_pretrained("/mnt/bn/ecom-govern-maxiangqian-lq/lj/Otter/exp_result/final_hfckpt", device_map="auto", **precision)
=======
>>>>>>> eb4623dc9986a12760b0167333c90c08f4e2609f


# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=32):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(vision_x, prompt: str, model=None, image_processor=None, tensor_dtype=None, batch_size=2) -> str:
    

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    # .unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


def get_test_video_path(root_dir, name):
    if name.startswith('H'):
        path = os.path.join(root_dir, 'test_humor', name)
    elif name.startswith('M'):
        path = os.path.join(root_dir, 'test_magic', name)
    else:
        path = os.path.join(root_dir, 'test_creative', name)
    return path

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/chengshuang/Otter/exp_result/otter9B_funqa_icl/final_hfckpt')
    parser.add_argument('--test_data_path', type=str, default='/data/chengshuang/Otter/data/annotation_with_ID/funqa_test_group_by_video.json')
    parser.add_argument('--output_path', type=str, default='/data/chengshuang/Otter/infer_data/test_res.jsonl')
    parser.add_argument('--video_path', type=str, default='/data/chengshuang/Otter/data/test')
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    model_path = args.model_path
    test_data_path = args.test_data_path
    output_path = args.output_path
    video_path = args.video_path
    batch_size = args.batch_size

    # This model version is trained on MIMIC-IT DC dataset.
    model = OtterForConditionalGeneration.from_pretrained(model_path, device_map="auto", **precision)
    # model = OtterForConditionalGeneration.from_pretrained("/mnt/bn/ecom-govern-maxiangqian-lq/lj/Otter/exp_result/final_hfckpt", device_map="auto", **precision)

    tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()
    
    # 读取/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/annotation_with_ID/funqa_test_group_by_video.json
    with open(test_data_path, 'r') as f:
        datas = json.load(f)
        
    for video_name, instructions in tqdm(datas.items(), total=len(datas)):
        video_url = get_test_video_path(video_path, video_name)
        print(video_url)
        frames_list = get_image(video_url)
        if isinstance(frames_list, Image.Image):
            vision_x = image_processor.preprocess([frames_list], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        elif isinstance(frames_list, list):  # list of video frames
            vision_x = image_processor.preprocess(frames_list, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")
        vision_x = vision_x
        for data in tqdm(instructions):
            prompts_input = data['instruction']
            task = data['task']
            if task == 'H1' or task == 'C1' or task == 'M1':
                data['predict'] = data['output']
            else:
                print(prompts_input)
                
                response = get_response(vision_x, prompts_input, model, image_processor, tensor_dtype, batch_size=batch_size)
                print(response)
                data['predict'] = response
                with open(output_path, 'a+') as f:
                    f.write(json.dumps(data) + '\n')