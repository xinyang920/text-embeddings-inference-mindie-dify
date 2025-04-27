from pathlib import Path
from typing import Optional

from loguru import logger
from transformers import AutoConfig
import torch

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.rerank_model import RerankModel
from ..utils.env import ENV
if ENV.backend == 'atb':
    import torch_npu
else:
    import mindietorch

__all__ = ["Model"]

# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)


def get_model(model_path: Path, dtype: Optional[str]):
    """Loads and returns the appropriate model class based on given path, data type and backend type.

    Args:
        model_path: The path to the pre-trained model.
        dtype: The data type of the model. If not specified or invalid, a RuntimeError will be raised.

    Returns:
        Model: An instance of a subclass of the `Model` class, which could either be:
            - RerankModel (if the architectures[0] in config.json ends with 'Classification')
            - FlashBert (if the model is based on BERT and CUDA is available)
            - DefaultModel (if all conditions above are not met)

    Raises:
        RuntimeError: If an unknown data type is provided for `dtype`.
        ValueError: If the device is CPU and the dtype is not `float32`.
    """
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    if ENV.device_id:
        if ENV.backend == 'atb':
            torch.npu.set_compile_mode(jit_compile=False)
            option = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceProd"}
            torch.npu.set_option(option)
            device = torch.device(f"npu:{int(ENV.device_id)}")
            torch.npu.set_device(torch.device(f"npu:{int(ENV.device_id)}"))
        else:
            mindietorch.set_device(int(ENV.device_id))
            device = torch.device(f"npu:{int(ENV.device_id)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if dtype != torch.float32:
            raise ValueError("CPU device only supports float32 dtype")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.architectures[0].endswith("Classification"):
        return RerankModel(model_path, device, dtype)
    else:
        if (
            config.model_type == "bert"
            and device.type == "cuda"
            and config.position_embedding_type == "absolute"
            and dtype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            return FlashBert(model_path, device, dtype)
        else:
            return DefaultModel(model_path, device, dtype)
    raise NotImplementedError