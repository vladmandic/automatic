from __future__ import annotations
from functools import partial
import re
import sys
import logging
import warnings
import urllib3
from modules import timer, errors

initialized = False
errors.install()
logging.getLogger("DeepSpeed").disabled = True

import torch # pylint: disable=C0411
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
timer.startup.record("transformers")

import onnxruntime # pylint: disable=W0611,C0411
onnxruntime.set_default_logger_severity(3)
timer.startup.record("onnx")

from fastapi import FastAPI # pylint: disable=W0611,C0411
import gradio # pylint: disable=W0611,C0411
timer.startup.record("gradio")
errors.install([gradio])

import pydantic # pylint: disable=W0611,C0411
timer.startup.record("pydantic")

import diffusers # pylint: disable=W0611,C0411
import diffusers.loaders.single_file # pylint: disable=W0611,C0411
logging.getLogger("diffusers.loaders.single_file").setLevel(logging.ERROR)
from tqdm.rich import tqdm # pylint: disable=W0611,C0411
diffusers.loaders.single_file.logging.tqdm = partial(tqdm, unit='C')
timer.startup.record("diffusers")

def get_packages():
    return {
        "torch": getattr(torch, "__long_version__", torch.__version__),
        "diffusers": diffusers.__version__,
        "gradio": gradio.__version__,
    }

errors.log.info(f'Load packages: {get_packages()}')

try:
    import os
    import math
    cores = os.cpu_count()
    affinity = len(os.sched_getaffinity(0))
    threads = torch.get_num_threads()
    if threads < (affinity / 2):
        torch.set_num_threads(math.floor(affinity / 2))
        threads = torch.get_num_threads()
        errors.log.debug(f'Detected: cores={cores} affinity={affinity} set threads={threads}')
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
