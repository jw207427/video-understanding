from transformers import AutoTokenizer, AutoModel
import torch
import utils
import math
from urllib.parse import urlparse
from typing import List, Optional
from djl_python import Input, Output
import boto3
import os
import tempfile

# Create an S3 client
s3 = boto3.client('s3')
_model = None
    
def parse_s3_url(s3_url):
    """
    Parse an S3 URL and return the bucket name and key.
    
    :param s3_url: The S3 URL in the format "s3://{bucket}/{key}"
    :return: A tuple containing the bucket name and key
    """
    parsed_url = urlparse(s3_url)
    
    if parsed_url.scheme != 's3':
        raise ValueError("Invalid S3 URL scheme. Expected 's3://'")
    
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    return bucket, key

def download_from_s3(s3_url):
    
    parsed_url = urlparse(s3_url)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    print(f"s3 bucket: {bucket}")
    print(f"key value: {key}")
    
    # Create a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()    
    
    try:
        # Download the file from S3 to the temporary file
        s3.download_file(bucket, key, tmp_path)
        print(f"File downloaded successfully to {tmp_path}")
        return tmp_path
    except Exception as e:
        print(f"Error downloading file from S3: {str(e)}")
        os.unlink(tmp_path)  # Remove the temporary file if download fails
        return None   
    
def split_model(model_name, gpu_count=1):
    device_map = {}
    world_size = gpu_count
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class Model():
    def __init__(self, properties):
        self.history = None
        
        self.get_model(properties)
        
        
    def get_model(self, properties):
        
        if "model_id" in properties:
            model_path = f'pretrained/{properties["model_id"]}'
            print(f"Load model path: {model_path} =======")
        else:
            # if not specified in the properties file, set to the smallest model
            model_path = "pretrained/InternVL2-1B"

        print("load tokenizer ===================")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        print("load model =======================")
        world_size = torch.cuda.device_count()
        if world_size > 1:
            # multi-gpu instance
            device_map = split_model(properties["model_id"])

            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map=device_map).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True).eval().cuda()
        
        print("InternVL model loaded ============")
       
    def generater_inference(self, inputs):
        
        data = inputs.get_as_json()
        
        pixel_values = None
        loaded_images = []
        loaded_videos = []
        num_patches_list = None
        video_prefix = ""
        if "images" in data:
            s3_images = data.pop("images")
            for idx, s3_url in enumerate(s3_images):
                file_path = download_from_s3(s3_url)
                # Load image and append to list
                img = utils.load_image(file_path, max_num=12).to(torch.bfloat16).cuda()
                loaded_images.append(img)

            # Concatenate all images
            pixel_values = torch.cat(loaded_images, dim=0)
            print(pixel_values)
        
        elif "video" in data:
            s3_video = data.pop("video")
            video_path = download_from_s3(s3_video)
            pixel_values, num_patches_list = utils.load_video(video_path, num_segments=8, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            
        if 'prompt' in data:
            # vide_prefix is empty until video is in the input data
            prompt = video_prefix+data.pop("prompt")
        else:
            prompt = 'Describe the images in detail'
        
        params = data["parameters"] if 'parameters' in data else {}
        
        # reset history
        if 'reset_history' in params:
            self.history = None
        
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, self.history = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config,
                                   num_patches_list=num_patches_list, history=self.history, return_history=True)
        
        return response
        
        
def handle(inputs: Input) -> Optional[Output]:
    global _model
    if not _model:
        _model = Model(inputs.get_properties())
        
    if inputs.is_empty():
        return None
    
    response = _model.generater_inference(inputs)

    return Output().add(response)