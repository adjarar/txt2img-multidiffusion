from upscale_utilities import *

def txt2img_openpose(url:str, prompts: json, upscaler: str, sampler: str, openpose_img: str,
                    output_dir:str, denoising_strength: float, steps: int, hr_steps: int,
                    scale_factor: int, batch_size: int, iterations: int):
    
    for id, prompt in enumerate(prompts):
        payload = {
            "enable_hr": True,
            "denoising_strength": denoising_strength,
            "steps": steps,
            "hr_second_pass_steps": hr_steps,
            "hr_scale": scale_factor,
            "batch_size": batch_size,
            "n_iter": iterations,
            "hr_upscaler": upscaler,
            "prompt": prompt,
            "sampler_name": sampler,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": openpose_img,
                            "model": "control_v11p_sd15_openpose [cab727d4]"
                        }
                    ]
                }
            }
        }

        response_json = response2json(url, 'txt2img', payload)

        for i, encoded_img in enumerate(response_json['images']):
            # this prevents saving the controlnet masks
            if i == batch_size * iterations:
                break

            decoded_img = decode_img(encoded_img)
            decoded_img.save(output_dir + '/' + str(id) + "_" + str(i) + '.png')
