from __future__ import annotations
from functools import partial
import os
import re
import sys
import logging
import warnings
import urllib3
from modules import timer, errors


initialized = False
errors.install()
logging.getLogger("DeepSpeed").disabled = True


# os.environ.setdefault('TORCH_LOGS', '-all')
import torch # pylint: disable=C0411
if torch.__version__.startswith('2.5.0'):
    errors.log.warning(f'Disabling cuDNN for SDP on torch={torch.__version__}')
    torch.backends.cuda.enable_cudnn_sdp(False)
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
    errors.log.debug(f'Load IPEX=={ipex.__version__}')
except Exception:
    pass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
import torchvision # pylint: disable=W0611,C0411
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them # pylint: disable=W0611,C0411
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
logging.getLogger("pytorch_lightning").disabled = True
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
try:
    torch._logging.set_logs(all=logging.ERROR, bytecode=False, aot_graphs=False, aot_joint_graph=False, ddp_graphs=False, graph=False, graph_code=False, graph_breaks=False, graph_sizes=False, guards=False, recompiles=False, recompiles_verbose=False, trace_source=False, trace_call=False, trace_bytecode=False, output_code=False, kernel_code=False, schedule=False, perf_hints=False, post_grad_graphs=False, onnx_diagnostics=False, fusion=False, overlap=False, export=None, modules=None, cudagraphs=False, sym_node=False, compiled_autograd_verbose=False) # pylint: disable=protected-access
except Exception:
    pass
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)
timer.startup.record("torch")

import transformers # pylint: disable=W0611,C0411
from transformers import logging as transformers_logging # pylint: disable=W0611,C0411
transformers_logging.set_verbosity_error()
timer.startup.record("transformers")

import accelerate # pylint: disable=W0611,C0411
timer.startup.record("accelerate")

import onnxruntime # pylint: disable=W0611,C0411
onnxruntime.set_default_logger_severity(3)
timer.startup.record("onnx")

from fastapi import FastAPI # pylint: disable=W0611,C0411
import gradio # pylint: disable=W0611,C0411
timer.startup.record("gradio")
errors.install([gradio])

import pydantic # pylint: disable=W0611,C0411
timer.startup.record("pydantic")

import diffusers.utils.import_utils # pylint: disable=W0611,C0411
diffusers.utils.import_utils._k_diffusion_available = True # pylint: disable=protected-access # monkey-patch since we use k-diffusion from git
diffusers.utils.import_utils._k_diffusion_version = '0.0.12' # pylint: disable=protected-access
import diffusers # pylint: disable=W0611,C0411
import diffusers.loaders.single_file # pylint: disable=W0611,C0411
import huggingface_hub # pylint: disable=W0611,C0411
logging.getLogger("diffusers.loaders.single_file").setLevel(logging.ERROR)
timer.startup.record("diffusers")


# patch different progress bars
import tqdm as tqdm_lib # pylint: disable=C0411
from tqdm.rich import tqdm # pylint: disable=W0611,C0411
diffusers.loaders.single_file.logging.tqdm = partial(tqdm, unit='C')

class _tqdm_cls():
    def __call__(self, *args, **kwargs):
        bar_format = 'Diffusers {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + '{desc}' + '\x1b[0m'
        return tqdm_lib.tqdm(*args, bar_format=bar_format, ncols=80, colour='#327fba', **kwargs)

class _tqdm_old(tqdm_lib.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs['bar_format'] = 'Diffusers {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + '{desc}' + '\x1b[0m'
        kwargs['ncols'] = 80
        super().__init__(*args, **kwargs)


transformers.utils.logging.tqdm = _tqdm_cls()
diffusers.pipelines.pipeline_utils.logging.tqdm = _tqdm_cls()
huggingface_hub._snapshot_download.hf_tqdm = _tqdm_old # pylint: disable=protected-access


def get_packages():
    return {
        "torch": getattr(torch, "__long_version__", torch.__version__),
        "diffusers": diffusers.__version__,
        "gradio": gradio.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
    }

try:
    import math
    cores = os.cpu_count()
    affinity = len(os.sched_getaffinity(0))
    threads = torch.get_num_threads()
    if threads < (affinity / 2):
        torch.set_num_threads(math.floor(affinity / 2))
        threads = torch.get_num_threads()
    errors.log.debug(f'System: cores={cores} affinity={affinity} threads={threads}')
except Exception:
    pass

try: # fix changed import in torchvision 0.17+, which breaks basicsr
    import torchvision.transforms.functional_tensor # pylint: disable=unused-import, ungrouped-imports
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

errors.log.info(f'Torch: torch=={torch.__version__} torchvision=={torchvision.__version__}')
errors.log.info(f'Packages: diffusers=={diffusers.__version__} transformers=={transformers.__version__} accelerate=={accelerate.__version__} gradio=={gradio.__version__}')
