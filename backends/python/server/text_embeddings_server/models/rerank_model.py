from pathlib import Path
from typing import Optional, Type, List

from opentelemetry import trace
from transformers import AutoModelForSequenceClassification
from loguru import logger
import torch

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Prediction, TokenEmbedding
from ..utils.env import ENV
if ENV.backend == 'atb':
    import torch_npu
else:
    import mindietorch

tracer = trace.get_tracer(__name__)


class RerankModel(Model):
    """
    RerankModel is a wrapper around a pre-trained reranker model that predicts similarity scores between given 
    query and documents.

    It extends the Model class and adds functionality to provide similrity score prediction according to the batch 
    of input data.
    """

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        """Initializes the RerankModel by loading the pre-trained model based on the given model path and backend.

        Args: 
            model_path: Path to the pre-trained model.
            device: The device to load the model on (e.g., NPU).
            dtype: The data type of the model's parameters (e.g., torch.float16).
        """

        if ENV.backend == 'atb':
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(device).eval()
        else:
            mindietorch.set_device(device.index)
            # 使用MindIE Torch后端时，请确保model_path（模型权重路径）下仅存在一个编译优化后的pt文件，否则请根据实际情况修改正则表达式和pt文件命名
            model = torch.jit.load(next(Path(model_path).rglob("*.pt"))).eval()  
        super(RerankModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        """Returns the class type of the batch that the reranker model expects.

        Returns:
            Type[PaddedBatch]: The type of the batch class expected by the reranker model. All tensors in this 
            batch are padded according to the longgest input sequece.
        """

        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> Optional[List[Embedding]]:
        """Logs an error indicating that the reranker model does not support the embed function."""
        logger.error("rerank model does not support embed function")

    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch) -> Optional[List[TokenEmbedding]]:
        """Logs an error indicating that the reranker model does not support the embed_all function."""
        logger.error("rerank model does not support embed_all function")

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Prediction]:
        """Predicts the similarity score for each query-text pair in the input batch.

        Args:
            batch: The input batch containing tokenized data (e.g., input_ids, attention_mask).

        Returns:
            List[Prediction]: A list of predictions, where each prediction represents the similarity score for 
            each query-text pair in the input batch.
        """

        kwargs = {"input_ids": batch.input_ids.to(self.device), "attention_mask": batch.attention_mask.to(self.device)}
        if ENV.backend == 'atb':
            scores = self.model(**kwargs, return_dict=True).logits.view(-1, ).float().tolist()
            return [
                Prediction(
                    values=[scores[i]]
                )
                for i in range(len(batch))
            ]
        else:
            scores = self.model(kwargs["input_ids"], kwargs["attention_mask"])[0].tolist()
            return [
                Prediction(
                    values=scores[i]
                )
                for i in range(len(batch))
            ]