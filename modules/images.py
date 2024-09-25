import io
import re
import os
import sys
import json
import queue
import random
import datetime
import threading
import numpy as np
import piexif
import piexif.helper
from PIL import Image, PngImagePlugin, ExifTags
from modules import sd_samplers, shared, script_callbacks, errors, paths
from modules.images_grid import image_grid, get_grid_size, split_grid, combine_grid, check_grid_size, get_font, draw_grid_annotations, draw_prompt_matrix, GridAnnotation, Grid # pylint: disable=unused-import
from modules.images_resize import resize_image # pylint: disable=unused-import
from modules.images_namegen import FilenameGenerator, get_next_sequence_number # pylint: disable=unused-import


debug = errors.log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None
try:
    from pi_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


def atomically_save_image():
    Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes
    while True:
        image, filename, extension, params, exifinfo, filename_txt = save_queue.get()
        with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            file.write(exifinfo)
        fn = filename + extension
        filename = filename.strip()
        if extension[0] != '.': # add dot if missing
            extension = '.' + extension
        try:
            image_format = Image.registered_extensions()[extension]
        except Exception:
            shared.log.warning(f'Save: unknown image format: {extension}')
            image_format = 'JPEG'
        if shared.opts.image_watermark_enabled or (shared.opts.image_watermark_position != 'none' and shared.opts.image_watermark_image != ''):
            image = set_watermark(image, shared.opts.image_watermark)
        exifinfo = (exifinfo or "") if shared.opts.image_metadata else ""
        # additional metadata saved in files
        if shared.opts.save_txt and len(exifinfo) > 0:
            try:
                with open(filename_txt, "w", encoding="utf8") as file:
                    file.write(f"{exifinfo}\n")
                shared.log.info(f'Save: text="{filename_txt}" len={len(exifinfo)}')
            except Exception as e:
                shared.log.warning(f'Save failed: description={filename_txt} {e}')
        # actual save
        if image_format == 'PNG':
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in params.pnginfo.items():
                pnginfo_data.add_text(k, str(v))
            save_args = { 'compress_level': 6, 'pnginfo': pnginfo_data if shared.opts.image_metadata else None }
        elif image_format == 'JPEG':
            if image.mode == 'RGBA':
                shared.log.warning('Save: removing alpha channel')
                image = image.convert("RGB")
            elif image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("L")
            save_args = { 'optimize': True, 'quality': shared.opts.jpeg_quality }
            if shared.opts.image_metadata:
                save_args['exif'] = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        elif image_format == 'WEBP':
            if image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
            save_args = { 'optimize': True, 'quality': shared.opts.jpeg_quality, 'lossless': shared.opts.webp_lossless }
            if shared.opts.image_metadata:
                save_args['exif'] = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
        else:
            save_args = { 'quality': shared.opts.jpeg_quality }
        try:
            image.save(fn, format=image_format, **save_args)
        except Exception as e:
            shared.log.error(f'Save failed: file="{fn}" format={image_format} args={save_args} {e}')
            errors.display(e, 'Image save')
        size = os.path.getsize(fn) if os.path.exists(fn) else 0
        shared.log.info(f'Save: image="{fn}" type={image_format} width={image.width} height={image.height} size={size}')
        if shared.opts.save_log_fn != '' and len(exifinfo) > 0:
            fn = os.path.join(paths.data_path, shared.opts.save_log_fn)
            if not fn.endswith('.json'):
                fn += '.json'
            entries = shared.readfile(fn, silent=True)
            idx = len(list(entries))
            if idx == 0:
                entries = []
            entry = { 'id': idx, 'filename': filename, 'time': datetime.datetime.now().isoformat(), 'info': exifinfo }
            entries.append(entry)
            shared.writefile(entries, fn, mode='w', silent=True)
            shared.log.info(f'Save: json="{fn}" records={len(entries)}')
        save_queue.task_done()


save_queue = queue.Queue()
save_thread = threading.Thread(target=atomically_save_image, daemon=True)
save_thread.start()


def save_image(image,
               path=None,
               basename='',
               seed=None,
               prompt=None,
               extension=shared.opts.samples_format,
               info=None,
               short_filename=False,
               no_prompt=False,
               grid=False,
               pnginfo_section_name='parameters',
               p=None,
               existing_info=None,
               forced_filename=None,
               suffix='',
               save_to_dirs=None,
            ): # pylint: disable=unused-argument
    debug(f'Save: fn={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access
    if image is None:
        shared.log.warning('Image is none')
        return None, None, None
    if not check_grid_size([image]):
        return None, None, None
    if path is None or path == '': # set default path to avoid errors when functions are triggered manually or via api and param is not set
        path = shared.opts.outdir_save
    namegen = FilenameGenerator(p, seed, prompt, image, grid=grid)
    suffix = suffix if suffix is not None else ''
    basename = basename if basename is not None else ''
    if shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        path = os.path.join(path, dirname)
    if forced_filename is None:
        if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0:
            file_decoration = shared.opts.samples_filename_pattern
        else:
            file_decoration = "[seq]-[prompt_words]"
        file_decoration = namegen.apply(file_decoration)
        file_decoration += suffix if suffix is not None else ''
        filename = os.path.join(path, f"{file_decoration}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{file_decoration}.{extension}")
    else:
        forced_filename += suffix if suffix is not None else ''
        filename = os.path.join(path, f"{forced_filename}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{forced_filename}.{extension}")
    pnginfo = existing_info or {}
    if info is None:
        info = image.info.get(pnginfo_section_name, '')
    if info is not None:
        pnginfo[pnginfo_section_name] = info
    params = script_callbacks.ImageSaveParams(image, p, filename, pnginfo)
    params.filename = namegen.sanitize(filename)
    dirname = os.path.dirname(params.filename)
    if dirname is not None and len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    params.filename = namegen.sequence(params.filename, dirname, basename)
    params.filename = namegen.sanitize(params.filename)
    # callbacks
    script_callbacks.before_image_saved_callback(params)
    exifinfo = params.pnginfo.get('UserComment', '')
    exifinfo = exifinfo + ', ' if len(exifinfo) > 0 else ''
    exifinfo += params.pnginfo.get(pnginfo_section_name, '')
    filename, extension = os.path.splitext(params.filename)
    filename_txt = f"{filename}.txt" if shared.opts.save_txt and len(exifinfo) > 0 else None
    save_queue.put((params.image, filename, extension, params, exifinfo, filename_txt)) # actual save is executed in a thread that polls data from queue
    save_queue.join()
    if not hasattr(params.image, 'already_saved_as'):
        debug(f'Image marked: "{params.filename}"')
        params.image.already_saved_as = params.filename
    script_callbacks.image_saved_callback(params)
    return params.filename, filename_txt, exifinfo


def save_video_atomic(images, filename, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3):
    try:
        import cv2
    except Exception as e:
        shared.log.error(f'Save video: cv2: {e}')
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if video_type.lower() == 'mp4':
        frames = images
        if interpolate > 0:
            try:
                import modules.rife
                frames = modules.rife.interpolate(images, count=interpolate, scale=scale, pad=pad, change=change)
            except Exception as e:
                shared.log.error(f'RIFE interpolation: {e}')
                errors.display(e, 'RIFE interpolation')
        video_frames = [np.array(frame) for frame in frames]
        fourcc = "mp4v"
        h, w, _c = video_frames[0].shape
        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*fourcc), fps=len(frames)/duration, frameSize=(w, h))
        for i in range(len(video_frames)):
            img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        size = os.path.getsize(filename)
        shared.log.info(f'Save video: file="{filename}" frames={len(frames)} duration={duration} fourcc={fourcc} size={size}')
    if video_type.lower() == 'gif' or video_type.lower() == 'png':
        append = images.copy()
        image = append.pop(0)
        if loop:
            append += append[::-1]
        frames=len(append) + 1
        image.save(
            filename,
            save_all = True,
            append_images = append,
            optimize = False,
            duration = 1000.0 * duration / frames,
            loop = 0 if loop else 1,
        )
        size = os.path.getsize(filename)
        shared.log.info(f'Save video: file="{filename}" frames={len(append) + 1} duration={duration} loop={loop} size={size}')


def save_video(p, images, filename = None, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3, sync: bool = False):
    if images is None or len(images) < 2 or video_type is None or video_type.lower() == 'none':
        return None
    image = images[0]
    if p is not None:
        seed = p.all_seeds[0] if getattr(p, 'all_seeds', None) is not None else p.seed
        prompt = p.all_prompts[0] if getattr(p, 'all_prompts', None) is not None else p.prompt
        namegen = FilenameGenerator(p, seed=seed, prompt=prompt, image=image)
    else:
        namegen = FilenameGenerator(None, seed=0, prompt='', image=image)
    if filename is None and p is not None:
        filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
        filename = os.path.join(shared.opts.outdir_video, filename)
        filename = namegen.sequence(filename, shared.opts.outdir_video, '')
    else:
        if os.pathsep not in filename:
            filename = os.path.join(shared.opts.outdir_video, filename)
    if not filename.lower().endswith(video_type.lower()):
        filename += f'.{video_type.lower()}'
    filename = namegen.sanitize(filename)
    if not sync:
        threading.Thread(target=save_video_atomic, args=(images, filename, video_type, duration, loop, interpolate, scale, pad, change)).start()
    else:
        save_video_atomic(images, filename, video_type, duration, loop, interpolate, scale, pad, change)
    return filename


def safe_decode_string(s: bytes):
    remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text # pylint: disable=unnecessary-lambda-assignment
    for encoding in ['utf-8', 'utf-16', 'ascii', 'latin_1', 'cp1252', 'cp437']: # try different encodings
        try:
            s = remove_prefix(s, b'UNICODE')
            s = remove_prefix(s, b'ASCII')
            s = remove_prefix(s, b'\x00')
            val = s.decode(encoding, errors="strict")
            val = re.sub(r'[\x00-\x09]', '', val).strip() # remove remaining special characters
            if len(val) == 0: # remove empty strings
                val = None
            return val
        except Exception:
            pass
    return None


def read_info_from_image(image: Image, watermark: bool = False):
    items = image.info or {}
    geninfo = items.pop('parameters', None) or items.pop('UserComment', None)
    if geninfo is not None and len(geninfo) > 0:
        if 'UserComment' in geninfo:
            geninfo = geninfo['UserComment']
        items['UserComment'] = geninfo

    if "exif" in items:
        try:
            exif = piexif.load(items["exif"])
        except Exception as e:
            shared.log.error(f'Error loading EXIF data: {e}')
            exif = {}
        for _key, subkey in exif.items():
            if isinstance(subkey, dict):
                for key, val in subkey.items():
                    if isinstance(val, bytes): # decode bytestring
                        val = safe_decode_string(val)
                    if isinstance(val, tuple) and isinstance(val[0], int) and isinstance(val[1], int) and val[1] > 0: # convert camera ratios
                        val = round(val[0] / val[1], 2)
                    if val is not None and key in ExifTags.TAGS: # add known tags
                        if ExifTags.TAGS[key] == 'UserComment': # add geninfo from UserComment
                            geninfo = val
                            items['parameters'] = val
                        else:
                            items[ExifTags.TAGS[key]] = val
                    elif val is not None and key in ExifTags.GPSTAGS:
                        items[ExifTags.GPSTAGS[key]] = val
    if watermark:
        wm = get_watermark(image)
        if wm != '':
            # geninfo += f' Watermark: {wm}'
            items['watermark'] = wm

    for key, val in items.items():
        if isinstance(val, bytes): # decode bytestring
            items[key] = safe_decode_string(val)

    for key in ['exif', 'ExifOffset', 'JpegIFOffset', 'JpegIFByteCount', 'ExifVersion', 'icc_profile', 'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'adobe', 'photoshop', 'loop', 'duration', 'dpi']: # remove unwanted tags
        items.pop(key, None)

    if items.get("Software", None) == "NovelAI":
        try:
            json_info = json.loads(items["Comment"])
            sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")
            geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
        except Exception as e:
            errors.display(e, 'novelai image parser')

    try:
        items['width'] = image.width
        items['height'] = image.height
        items['mode'] = image.mode
    except Exception:
        pass

    return geninfo, items


def image_data(data):
    import gradio as gr
    if data is None:
        return gr.update(), None
    err1 = None
    err2 = None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        info, _ = read_info_from_image(image)
        errors.log.debug(f'Decoded object: image={image} metadata={info}')
        return info, None
    except Exception as e:
        err1 = e
    try:
        if len(data) > 1024 * 10:
            errors.log.warning(f'Error decoding object: data too long: {len(data)}')
            return gr.update(), None
        info = data.decode('utf8')
        errors.log.debug(f'Decoded object: data={len(data)} metadata={info}')
        return info, None
    except Exception as e:
        err2 = e
    errors.log.error(f'Error decoding object: {err1 or err2}')
    return gr.update(), None


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""
    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background
    return img.convert('RGB')


def set_watermark(image, watermark):
    if shared.opts.image_watermark_position != 'none': # visible watermark
        wm_image = None
        try:
            wm_image = Image.open(shared.opts.image_watermark_image)
        except Exception as e:
            shared.log.warning(f'Set image watermark: fn="{shared.opts.image_watermark_image}" {e}')
        if wm_image is not None:
            if shared.opts.image_watermark_position == 'top/left':
                position = (0, 0)
            elif shared.opts.image_watermark_position == 'top/right':
                position = (image.width - wm_image.width, 0)
            elif shared.opts.image_watermark_position == 'bottom/left':
                position = (0, image.height - wm_image.height)
            elif shared.opts.image_watermark_position == 'bottom/right':
                position = (image.width - wm_image.width, image.height - wm_image.height)
            elif shared.opts.image_watermark_position == 'center':
                position = ((image.width - wm_image.width) // 2, (image.height - wm_image.height) // 2)
            else:
                position = (random.randint(0, image.width - wm_image.width), random.randint(0, image.height - wm_image.height))
            try:
                for x in range(wm_image.width):
                    for y in range(wm_image.height):
                        r, g, b, _a = wm_image.getpixel((x, y))
                        if not (r == 0 and g == 0 and b == 0):
                            image.putpixel((x+position[0], y+position[1]), (r, g, b))
                shared.log.debug(f'Set image watermark: fn="{shared.opts.image_watermark_image}" image={wm_image} position={position}')
            except Exception as e:
                shared.log.warning(f'Set image watermark: image={wm_image} {e}')

    if shared.opts.image_watermark_enabled: # invisible watermark
        from imwatermark import WatermarkEncoder
        wm_type = 'bytes'
        wm_method = 'dwtDctSvd'
        wm_length = 32
        length = wm_length // 8
        info = image.info
        data = np.asarray(image)
        encoder = WatermarkEncoder()
        text = f"{watermark:<{length}}"[:length]
        bytearr = text.encode(encoding='ascii', errors='ignore')
        try:
            encoder.set_watermark(wm_type, bytearr)
            encoded = encoder.encode(data, wm_method)
            image = Image.fromarray(encoded)
            image.info = info
            shared.log.debug(f'Set invisible watermark: {watermark} method={wm_method} bits={wm_length}')
        except Exception as e:
            shared.log.warning(f'Set invisible watermark error: {watermark} method={wm_method} bits={wm_length} {e}')

    return image


def get_watermark(image):
    from imwatermark import WatermarkDecoder
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = 32
    data = np.asarray(image)
    decoder = WatermarkDecoder(wm_type, wm_length)
    try:
        decoded = decoder.decode(data, wm_method)
        wm = decoded.decode(encoding='ascii', errors='ignore')
    except Exception:
        wm = ''
    return wm
