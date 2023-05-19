import json
import upscalers
import samplers
from upscale_utilities import *
from txt2img_openpose import txt2img_openpose
from multidiffusion_upscale import multidiffusion_upscale
import discord
import requests

url = "http://127.0.0.1:7860"
work_dir = "/home/webui"
output_dir_txt2img = work_dir + "/txt2img_output"
output_dir_img2img = work_dir + "/img2img_output"


multidiffusion_upscale(url, output_dir_txt2img, output_dir_img2img, samplers.euler_a, upscalers.ultrasharp,
                       denoising_strength=0.3, steps=30, latent_batch_size=8, scale_factor=2)

webhook = discord.SyncWebhook.partial(1108891310351470662, '5Q-A_WqDX7Iiu6Y30oyifxGHdfL2PeErrW0MWA5kFjRTcGXbMv_Sv6NmtXhIwiOX0hf_')
webhook.send('Finnished upscaling images', username='Multidiffusion Upcaler')