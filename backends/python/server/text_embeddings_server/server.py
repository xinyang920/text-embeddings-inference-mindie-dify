import asyncio
from pathlib import Path
from typing import Optional

from grpc import aio
from loguru import logger
from grpc_reflection.v1alpha import reflection
import torch

from text_embeddings_server.models import Model, get_model
from text_embeddings_server.pb import embed_pb2_grpc, embed_pb2
from text_embeddings_server.utils.tracing import UDSOpenTelemetryAioServerInterceptor
from text_embeddings_server.utils.interceptor import ExceptionInterceptor
from .utils.env import ENV
if ENV.backend == 'atb':
    import torch_npu
else:
    import mindietorch


class EmbeddingService(embed_pb2_grpc.EmbeddingServiceServicer):
    """Handles gRPC requests for the text embeddings service.

    This class implements the EmbeddingServiceServicer interface, providing methods for embedding, prediction 
    and health check functionalities.

    Attributes:
        model (Model): The model used for generating embeddings and similarity scores.
        _inference_mode_raii_guard (torch._C._InferenceMode): A context manager to enforce inference mode.
    """

    def __init__(self, model: Model):
        """Initializes the EmbeddingService with a model.

        Args:
            model (Model): The model to use for embedding or reranker service.
        """

        self.model = model
        # Force inference mode for the lifetime of EmbeddingService
        self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Health(self, request, context):
        """Handles the health check request.

        Verifies if the model is able to run on the provided device.

        Args:
            request: The gRPC health check request.
            context: The gRPC context for the request.

        Returns:
            embed_pb2.HealthResponse: A response indicating the service health status.
        """

        if self.model.device.type == "npu":
            health_check_tensor = torch.zeros((2, 2))
            health_check_tensor.to(self.model.device)
        return embed_pb2.HealthResponse()

    async def Embed(self, request, context):
        """Handles the embed request.

        Processes the incoming request, generates embeddings using the provided model, and returns the hidden 
        state of the first token for each sequence in the batch.

        Args:
            request: The gRPC request containing a batch of data for embedding.
            context: The gRPC context for the request.

        Returns:
            embed_pb2.EmbedResponse: A response containing generated the sliced embeddings for the request.
        """

        batch = self.model.batch_type.from_pb(request, self.model.device)

        embeddings = self.model.embed(batch)

        return embed_pb2.EmbedResponse(embeddings=embeddings)

    async def Embed_all(self, request, context):
        """Handles the embed_all request.

        Processes the incoming request, generates embeddings using the provided model, and returns hidden states 
        of all tokens for each sequence in the batch.

        Args:
            request: The gRPC request containing a batch of data for embedding.
            context: The gRPC context for the request.

        Returns:
            embed_pb2.RawEmbedResponse: A response containing all the embeddings for the request.
        """

        batch = self.model.batch_type.from_pb(request, self.model.device)

        embeddings = self.model.embed_all(batch)

        return embed_pb2.RawEmbedResponse(allembeddings=embeddings)

    async def Predict(self, request, context):
        """Handles the predict request.

        Processes the incoming request, generates predictions using the provided model, and returns final scores.

        Args:
            request: The gRPC request containing a batch of data for scores prediction.
            context: The gRPC context for the request.

        Returns:
            embed_pb2.PredictResponse: A response containing the model's predictions for the request.
        """

        batch = self.model.batch_type.from_pb(request, self.model.device)

        predictions = self.model.predict(batch)

        return embed_pb2.PredictResponse(predictions=predictions)


def serve(
    model_path: Path,
    dtype: Optional[str],
    uds_path: Path,
):
    """Starts the gRPC server and serves the text embedding service.

    This function initializes the model and starts a server that listens for incoming requests to generate embeddings, 
    predictions or check health.

    Args:
        model_path: Path to the model directory.
        dtype: Data type for model initialization.
        uds_path: Path to the Unix Domain Socket for the server to listen on.
    """

    async def serve_inner(
        model_path: Path,
        dtype: Optional[str] = None,
    ):
        """The inner asynchronous function to run the gRPC server.

        Args:
            model_path: Path to the model directory.
            dtype: Data type for model initialization.

        Raises:
            Exception: If there is an error during model initialization.
        """

        unix_socket = f"unix://{uds_path}"

        try:
            model = get_model(model_path, dtype)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        embed_pb2_grpc.add_EmbeddingServiceServicer_to_server(
            EmbeddingService(model), server
        )
        SERVICE_NAMES = (
            embed_pb2.DESCRIPTOR.services_by_name["EmbeddingService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(unix_socket)

        await server.start()

        logger.info(f"Server started at {unix_socket}")

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(model_path, dtype))