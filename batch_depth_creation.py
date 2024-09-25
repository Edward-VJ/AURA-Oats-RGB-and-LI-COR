# Copyright (c) 2024, Edward Jakunskas, Department of Electronic Engineering, Maynooth University
#
# Maintainer: Edward Jakunskas (edward.jakunskas@mu.ie)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)

# load Depth anything model vitl
encoder = 'vitl'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/media/hdd/Workspace/oat_paper/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=False))
model = model.to(DEVICE).eval()

greenhouse = "GH4"
daily_dir = "chosen_img"
base_dir = os.path.join("/run/user/1000/gvfs/smb-share:server=149.157.140.139,share=public", greenhouse)
input_directory = os.path.join(base_dir, daily_dir)
depth_output_directory = input_directory + "_depth_raw"

os.makedirs(depth_output_directory, exist_ok=True)

#camera ID and digit selection dictionary, selecting focus distance 49 for cam 1844...F00 etc
depth_select = {
    "18443010715BEE0F00": "49",
    "19443010D1BE671300": "60",
}

#filter files based on camera ID and the digits before the extension
file_list = []

file_list_temp = [f for f in os.listdir(input_directory) if f.endswith('.png') or f.endswith('.jpg')]

for file in file_list_temp:
    # exctract cam ID
    camera_id = file.split('cam', 1)[1].split('_')[0]
    if camera_id in depth_select:
        digits_before_ext = file.split('_')[-1].split('.')[0]  #get the part before .png/.jpg
        if digits_before_ext == depth_select[camera_id]:
            file_list.append(file)


for file in file_list:
    img_path = os.path.join(input_directory, file)
    raw_img = cv2.imread(img_path)
    
    # run the model to compute depth
    depth_map = model.infer_image(raw_img)
    
    # save the depth map
    depth_output_path = os.path.join(depth_output_directory, f"depth_{file.rsplit('.', 1)[0]}.jpg") #save as jpg because had problems with png in SAM.
    cv2.imwrite(depth_output_path, depth_map)

print("Depth processing complete.")
