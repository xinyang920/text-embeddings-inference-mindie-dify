from abc import ABC, abstractmethod
from dataclasses import dataclass
from opentelemetry import trace
import torch

from text_embeddings_server.pb import embed_pb2
from text_embeddings_server.pb.embed_pb2 import Embedding, Prediction, TokenEmbedding

from ..utils.env import ENV
if ENV.backend == 'atb':
    import torch_npu
else:
    import mindietorch

tracer = trace.get_tracer(__name__)


class Batch(ABC):
    """Abstract base class for batching input data for embedding and prediction.

    This class provides an interface for batching input data and converting it from protocol buffer
    format (EmbedRequest) into a format suitable for the model.

    Methods:
        from_pb(pb, device): Converts a protocol buffer format EmbedRequest to a Batch instance.
        __len__(): Returns the number of sequences in the batch.
    """

    @classmethod
    @abstractmethod
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "Batch":
        """Converts an EmbedRequest protocol buffer format to a Batch object.

        Args:
            pb: The protocol buffer message containing input data.
            device: The device to allocate tensors.

        Returns:
            Batch: A new instance of a batch (either PaddedBatch or FlashBatch).
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Returns the number of sequences in the batch."""
        raise NotImplementedError


@dataclass
class PaddedBatch(Batch):
    """Represents a batch of padded input data.

    This class holds a batch of input sequences, padding them to the same length. The data is represented
    as tensors, and all sequences are padded to the `max_length`.

    Attributes:
        input_ids: Tensor containing input token IDs for each sequence in the batch.
        token_type_ids: Tensor containing token type IDs for each sequence.
        position_ids: Tensor containing position IDs for each sequence.
        attention_mask: Tensor for the attention mask, indicating valid tokens in each sequence.
        max_length: The maximum sequence length for padding.

    Methods:
        from_pb(pb, device): Converts a protocol buffer format EmbedRequest to a PaddedBatch instance.
        __len__(): Returns the number of sequences in the batch.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    max_length: int

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "PaddedBatch":
        """Converts an EmbedRequest protocol buffer format to a PaddedBatch instance.

        Args:
            pb: The protocol buffer message containing input data.
            device: The device to allocate tensors.

        Returns:
            PaddedBatch: A PaddedBatch instance containing padded input data.
        """

        # Allocate padded tensors all at once
        all_tensors = torch.zeros(
            [4, len(pb.cu_seq_lengths) - 1, pb.max_length], dtype=torch.int32, device='cpu'
        )
        max_length=pb.max_length

        for i, start_index in enumerate(pb.cu_seq_lengths[:-1]):
            end_index = pb.cu_seq_lengths[i + 1]
            input_length = end_index - start_index

            all_tensors[0, i, :input_length] = torch.tensor(
                pb.input_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[1, i, :input_length] = torch.tensor(
                pb.token_type_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[2, i, :input_length] = torch.tensor(
                pb.position_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[3, i, :input_length] = 1
        """
        # Move padded tensors all at once
        all_tensors = all_tensors.to(device)
        """
        return PaddedBatch(
            input_ids=all_tensors[0],
            token_type_ids=all_tensors[1],
            position_ids=all_tensors[2],
            attention_mask=all_tensors[3],
            max_length=max_length,
        )

    def __len__(self):
        """Returns the number of sequences in the batch."""
        return len(self.input_ids)


@dataclass
class FlashBatch(Batch):
    """Represents a batch of input data for flash inference.

    This class is used for models that support flash-based batching, where input data is packed into
    larger sequences for efficient processing.

    Attributes:
        input_ids: Tensor containing input token IDs for each sequence in the batch.
        token_type_ids: Tensor containing token type IDs for each sequence.
        position_ids: Tensor containing position IDs for each sequence.
        cu_seqlens: Tensor containing sequence lengths for each batch element.
        max_s: The maximum sequence length for this batch.
        size: The number of sequences in the batch.

    Methods:
        from_pb(pb, device): Converts a protocol buffer format EmbedRequest to a FlashBatch instance.
        __len__(): Returns the number of sequences in the batch.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_s: int
    size: int

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "FlashBatch":
        """Converts an EmbedRequest protocol buffer format to a FlashBatch instance.

        Args:
            pb: The protocol buffer message containing input data.
            device: The device to allocate tensors.

        Returns:
            FlashBatch: A FlashBatch instance containing input data for a FlashBert model.

        Raises:
            RuntimeError: If the device is not 'cuda', FlashBatch is not supported.
        """
        if device.type != "cuda":
            raise RuntimeError(f"FlashBatch does not support device {device}")

        batch_input_ids = torch.tensor(pb.input_ids, dtype=torch.int32, device=device)
        batch_token_type_ids = torch.tensor(
            pb.token_type_ids, dtype=torch.int32, device=device
        )
        batch_position_ids = torch.tensor(
            pb.position_ids, dtype=torch.int32, device=device
        )

        cu_seqlens = torch.tensor(pb.cu_seq_lengths, dtype=torch.int32, device=device)

        return FlashBatch(
            input_ids=batch_input_ids,
            token_type_ids=batch_token_type_ids,
            position_ids=batch_position_ids,
            cu_seqlens=cu_seqlens,
            max_s=pb.max_length,
            size=len(cu_seqlens) - 1,
        )

    def __len__(self):
        """Returns the number of sequences in the batch."""
        return self.size