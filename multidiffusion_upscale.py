from upscale_utilities import *

def multidiffusion_upscale(url: str, input_dir: str, output_dir: str, sampler: str, upscaler: str,
                           denoising_strength: float, steps: int, latent_batch_size: int, scale_factor: int):
    
    for i, img_path in enumerate(glob.glob(f'{input_dir}/*')):
        payload = {
            "init_images": [
                encode_img(img_path)
            ],
            "sampler_name": sampler,
            "denoising_strength": denoising_strength,
            "steps": steps,
            "alwayson_scripts": {
                "Tiled Diffusion": {
                    "args": [
                        True,
                        "MultiDiffusion",
                        False, True,
                        None, None,
                        64, 64, 12,
                        latent_batch_size,
                        upscaler,
                        scale_factor              
                    ]
                }
            }
        }

        response = response2json(url, "img2img", payload)

        decoded_img = decode_img(response['images'][0])
        decoded_img.save(output_dir + '/' + str(i) + '.png')
