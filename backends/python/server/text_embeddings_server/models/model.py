from abc import ABC, abstractmethod
from typing import List, TypeVar, Type

import torch
from loguru import logger

from text_embeddings_server.models.types import Batch, Embedding, Prediction, TokenEmbedding
from ..utils.env import ENV
if ENV.backend == 'atb':
    import torch_npu
    from atb_llm.utils.initial import NPUSocInfo

B = TypeVar("B", bound=Batch)


class Model(ABC):
    """Abstract base class for all model types in TEI-MindIE.

    This class defines the common interface and properties for all models used in TEI-MindIE, including 
    methods for embedding generation and similarity score prediction.

    Attributes:
        model: The model object used for embedding generation or similarity score prediction.
        device: The device to load the model on (e.g., NPU).
        dtype: The data type of the model's parameters (e.g., torch.float16).
    """

    def __init__(
        self,
        model,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initializes the Model class based on the given model object, device and dtype. For ATB backend,
        this function also execute npu_cormat_cast on the model's named_modules according to NPU's SoC 
        information.  

        Args:
            model: The model object used for embedding generation or similarity score prediction.
            device: The device to load the model on (e.g., NPU).
            dtype: The data type of the model's parameters (e.g., torch.float16).
        """

        self.model = model
        self.dtype = dtype
        self.device = device
        if ENV.backend == 'atb':
            def trans_data(model, soc_info):
                if not soc_info.need_nz:
                    for _, module in model.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
                    logger.info(f"soc info: {soc_info.soc_version}, support ND")
                else:
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            if name == 'lm_head':
                                # eliminate TransData op before lm_head calculation
                                module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
                    logger.info(f"soc info: {soc_info.soc_version}, support NZ")

                for _, module in model.named_modules():
                    if isinstance(module, torch.nn.Embedding):
                        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
            self.soc_info = NPUSocInfo()
            trans_data(self.model, self.soc_info)

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, batch: B) -> List[Embedding]:
        raise NotImplementedError

    @abstractmethod
    def embed_all(self, batch: B) -> List[TokenEmbedding]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: B) -> List[Prediction]:
        raise NotImplementedError