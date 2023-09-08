from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer
text_tokenizer = LlamaTokenizer.from_pretrained('luodian/llama-7b-hf',cache_dir='/lustre/S/zhangyang/chengshuang/LLM/Otter/OTTER-Video-LLaMA7B-DenseCaption')


# import mimetypes
# import os
# from typing import Union
# import cv2
# import requests
# import torch
# import transformers
# from PIL import Image
# import sys

# # make sure you can properly access the otter folder
# from otter.modeling_otter import OtterForConditionalGeneration


# # This model version is trained on MIMIC-IT DC dataset.
# model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", cache_dir='/lustre/S/zhangyang/chengshuang/LLM/Otter/model_llm')
# model.text_tokenizer.padding_side = "left"
# tokenizer = model.text_tokenizer
# image_processor = transformers.CLIPImageProcessor()
# model.eval()