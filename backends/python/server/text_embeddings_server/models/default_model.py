from pathlib import Path
from typing import Optional, Type, List

from opentelemetry import trace
from transformers import AutoModel, AutoConfig
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


class DefaultModel(Model):
    """DefaultModel is a wrapper around a pre-trained embedding model that generates embeddings for input documents.

    It extends the Model class and adds functionality to provide embedding generation according to the batch of 
    input data.

    Attributes:
        config: The model configuration about the given pre-trained model.
        hidden_size: The size of the model's hidden states.
        is_causal: A flag indicating whether the embedding model is a causal model whose `architecture` ends with 
        "CausalLM". If True, embed function will return the embedding of the last token; otherwise, it returns the 
        embedding of the first token.
    """

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        """ 
        Initializes the DefaultModel by loading the pre-trained model based on the given model path and backend 
        type, while also setting necessary member variables for embedding generation.

        Args: 
            model_path: Path to the pre-trained model.
            device: The device to load the model on (e.g., NPU).
            dtype: The data type of the model's parameters (e.g., torch.float16).
        """

        self.config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        self.hidden_size = self.config.hidden_size
        self.is_causal = self.config.architectures[0].endswith('CausalLM')

        if ENV.backend == 'atb':
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(device).eval()
        else:
            mindietorch.set_device(device.index)
              # 使用MindIE Torch后端时，请确保model_path（模型权重路径）下仅存在一个编译优化后的pt文件，否则请根据实际情况修改正则表达式和pt文件命名
            model = torch.jit.load(next(Path(model_path).rglob("*.pt"))).eval().to(device)
        super(DefaultModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        """Returns the class type of the batch that the embedding model expects.

        Returns:
            Type[PaddedBatch]: The type of the batch class expected by the embedding model. All tensors in this 
            batch are padded according to the longgest input sequece.
        """

        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        """
        Generates embeddings for a given batch of input data and returns the hidden state of the first or last 
        token for each input sequence in the batch.

        Args:
            batch: The input batch containing tokenized data (e.g., input_ids, attention_mask).

        Returns:
            List[Embedding]: A list of embeddings, where each embedding represents the hidden state of the first 
            or last token for each input sequence in the batch.
        """

        kwargs = {"input_ids": batch.input_ids.to(self.device), "attention_mask": batch.attention_mask.to(self.device)}
        if ENV.backend == 'atb':
            output = self.model(**kwargs, return_dict=True)
            embedding = output[0]
        else:
            output = self.model(kwargs["input_ids"], kwargs["attention_mask"])
            if isinstance(output, dict):
                embedding = output['last_hidden_state'].to('cpu')
            else:
                embedding = output[0].to('cpu')

        if self.is_causal:
            # For causal models, get the embedding of the last token
            embedding = embedding[:, -1]
        else:
            # For non-causal models, get the embedding of the first token
            embedding = embedding[:, 0]

        cpu_results = embedding.contiguous().view(-1).tolist()
        return [
            Embedding(
                values=cpu_results[i * self.hidden_size: (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch) -> List[TokenEmbedding]:
        """
        Generates embeddings for a given batch of input data and returns hidden states of all tokens for each 
        input sequence in the batch.

        Args:
            batch: The input batch containing tokenized data (e.g., input_ids, attention_mask).

        Returns:
            List[TokenEmbedding]: A list of token embeddings, where each token embedding represents hidden states 
            of all tokens for each input sequence in the batch.
        """

        kwargs = {"input_ids": batch.input_ids.to(self.device), "attention_mask": batch.attention_mask.to(self.device)}
        if ENV.backend == 'atb':
            output = self.model(**kwargs, return_dict=True)
            embedding = output[0]
        else:
            output = self.model(kwargs["input_ids"], kwargs["attention_mask"])
            if isinstance(output, dict):
                embedding = output['last_hidden_state'].to('cpu')
            else:
                embedding = output[0].to('cpu')

        cpu_results = embedding.contiguous().view(-1).tolist()
        embedding_result = []
        for i in range(len(batch)):
            base_index = i * batch.max_length * self.hidden_size
            tmp_embedding = [
                Embedding(values=cpu_results[
                                 base_index + j * self.hidden_size: base_index + (j + 1) * self.hidden_size
                                 ])
                for j in range(batch.input_ids.size(1))
            ]
            token_embeddings = TokenEmbedding(embeddings=tmp_embedding)
            embedding_result.append(token_embeddings)

        return embedding_result

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> Optional[List[Prediction]]:
        """Logs an error indicating that the embedding model does not support the predict function."""
        logger.error("embedding model does not support predict function")