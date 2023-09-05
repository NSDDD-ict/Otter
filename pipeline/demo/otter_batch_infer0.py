import mimetypes
import os
from typing import Union
import cv2
import numpy as np
import requests
import torch
import transformers
from PIL import Image
import sys
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"   #

sys.path.append("/home/hnu2/.ss/Otter")
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

# This model version is trained on MIMIC-IT DC dataset.
# model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", device_map="auto", **precision)
# model = OtterForConditionalGeneration.from_pretrained("/mnt/bn/ecom-govern-maxiangqian-lq/lj/Otter/exp_result/final_hfckpt", device_map="auto", **precision)
# model = OtterForConditionalGeneration.from_pretrained("/home/hnu2/.ss/Otter/saved_model/Otter_trans", device_map="auto", **precision)
model = OtterForConditionalGeneration.from_pretrained("/mnt/bn/ecom-govern-maxiangqian-lq/lj/OTTER-9B-INIT", device_map="auto", **precision)

tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()

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
    if len(prompt)> 1:
        return [str("<image>User: {} GPT:<answer>".format(i)) for i in prompt]
    else:
        return f"<image>User: {prompt} GPT:<answer>"


def get_response(vision_x, prompt: str, model=None, image_processor=None, tensor_dtype=None, batch_size=2) -> str:
    

    lang_x = model.text_tokenizer(
        get_formatted_prompt(prompt),
        return_tensors="pt",
        padding=True,
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    # .unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    if batch_size > 1:
        vision_x = vision_x.repeat_interleave(batch_size, dim=0).to(dtype=model_dtype)
    else:
        vision_x = vision_x.to(dtype=model_dtype)

    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    # generated_text = model.generate(
    #     vision_x=vision_x.to(model.device),
    #     lang_x=lang_x_input_ids.to(model.device),
    #     attention_mask=lang_x_attention_mask.to(model.device),
    #     max_new_tokens=512,
    #     num_beams=3,
    #     no_repeat_ngram_size=3,
    #     bad_words_ids=bad_words_id,
    # )
    # tokenizer.eos_token_id = 50277
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512, ## 有长度限制
        # num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id, ## TODO 同demo代码，尚未研究作用
        pad_token_id=32003 # 去掉会警告 The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results. Setting `pad_token_id` to `eos_token_id`:50277 for open-end generation.
    )

    if batch_size > 1:
        parsed_output = [(
            model.text_tokenizer.decode(generated_text[i])
                .split("<answer>")[-1]
                .lstrip()
                .rstrip()
                .split("<|endofchunk|>")[0]
                .lstrip()
                .rstrip()
                .lstrip('"')
                .rstrip('"')
        ) for i in range(batch_size)]
    else:
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

# 读取/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/annotation_with_ID/funqa_test_group_by_video.json
with open('/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/annotation_with_ID/funqa_test_group_by_video.json', 'r') as f:
    datas = json.load(f)

for video_name, instructions in tqdm(datas.items(), total=len(datas)):
    video_url = get_test_video_path('/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/test', video_name)
    print(video_url)
    frames_list = get_image(video_url)
    if isinstance(frames_list, Image.Image):
        vision_x = image_processor.preprocess([frames_list], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(frames_list, list):  # list of video frames
        vision_x = image_processor.preprocess(frames_list, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")
    batch_size = 2
    # vision_x = vision_x

    assert len(instructions) == 20 ## 确保一定为20 且前五 作废

    for data in instructions[:5]:
        task = data['task']
        if task == 'H1' or task == 'C1' or task == 'M1':
            data['predict'] = data['output']

    test_list = np.array_split(list(range(5,20)), round( 15 / batch_size))
    for list_ in test_list:
        prompts_inputs = [ instructions[j]['instruction'] for j in list_]
        response = get_response(vision_x, prompts_inputs, model, image_processor, tensor_dtype, batch_size=batch_size)
        print(response)
        if batch_size == 1:
            data['predict'] = response
            with open('/mnt/bn/ecom-govern-maxiangqian-lq/lj/Otter/infer_data/test_res_batch.jsonl', 'a+') as f:
                f.write(json.dumps(data) + '\n')
        else:
            ## 一次性写bsz个数据
            for ind, jj in enumerate(list_):
                instructions[jj]['predict'] = response[ind]
                with open('/mnt/bn/ecom-govern-maxiangqian-lq/lj/Otter/infer_data/test_res_batch.jsonl', 'a+') as f:
                    f.write(json.dumps(instructions[jj]) + '\n')
