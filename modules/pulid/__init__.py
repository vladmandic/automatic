"""
Credit and original implementation: <https://github.com/ToTheBeginning/PuLID>
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))
from pipe_sdxl import PuLIDPipeline as PuLIDPipelineXL
from pulid_utils import resize_numpy_image_long as resize
import attention_processor as attention
