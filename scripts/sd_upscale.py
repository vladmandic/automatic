import math
import gradio as gr
from PIL import Image
import modules.scripts as scripts
from modules import processing, shared, images, devices
from modules.processing import Processed
from modules.shared import opts, state, log


class Script(scripts.Script):
    def title(self):
        return "SD Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row():
            info = gr.HTML("<span>&nbsp SD Upscale</span><br>")
        with gr.Row():
            overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, elem_id=self.elem_id("overlap"))
            scale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label='Scale Factor', value=2.0, elem_id=self.elem_id("scale_factor"))
        with gr.Row():
            upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", elem_id=self.elem_id("upscaler_index"))
        return [info, overlap, upscaler_index, scale_factor]

    def run(self, p, _, overlap, upscaler_index, scale_factor): # pylint: disable=arguments-differ
        init_img = None
        if hasattr(p, 'init_images') and p.init_images is not None:
            init_img = p.init_images[0]
        elif hasattr(p.task_args, 'image') and p.task_args['image'] is not None:
            init_img = p.task_args['image'][0]

        if init_img is None:
            return None
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]
        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name
        initial_info = None
        seed = p.seed
        if upscaler.name != "None":
            img = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
        else:
            img = init_img
        devices.torch_gc()
        grid = images.split_grid(img, tile_w=init_img.width, tile_h=init_img.height, overlap=overlap)
        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        work = []
        for _y, _h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / batch_size)
        state.job_count = batch_count * upscale_count
        log.info(f"SD upscale: images={len(work)} tiles={len(grid.tiles)} batches={state.job_count}")

        result_images = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed
            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i * batch_size:(i + 1) * batch_size]
                state.job = f"upscale batch {i+1+n*batch_count}/{state.job_count}"
                processed = processing.process_images(p)
                if initial_info is None:
                    initial_info = processed.info
                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for _y, _h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

        processed = Processed(p, result_images, seed, initial_info)
        log.info(f"SD upscale: images={result_images}")
        return processed
