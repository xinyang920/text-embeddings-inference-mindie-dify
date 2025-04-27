# Text Embeddings Inference

<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models.

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on an Nvidia A10 with a sequence
length of 512 tokens:

<p>
  <img src="assets/bs1-lat.png" width="400" />
  <img src="assets/bs1-tp.png" width="400" />
</p>
<p>
  <img src="assets/bs32-lat.png" width="400" />
  <img src="assets/bs32-tp.png" width="400" />
</p>

</div>

## Table of contents

- [Text Embeddings Inference](#text-embeddings-inference)
- [Text Embeddings Inference](#text-embeddings-inference-1)
  - [Table of contents](#table-of-contents)
    - [Using Re-rankers models](#using-re-rankers-models)
  - [Hardware Support](#hardware-support)
    - [Huawei NPU](#huawei-npu)
  - [API 格式变更](#api-格式变更)
    - [Rerank 接口适配](#rerank-接口适配)

Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence
classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding,
Ember, GTE and E5. TEI implements many features such as:

* No model graph compilation step
* Metal support for local execution on Macs
* Small docker images and fast boot times. Get ready for true serverless!
* Token based dynamic batching
* Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
  [Candle](https://github.com/huggingface/candle)
  and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
* [Safetensors](https://github.com/huggingface/safetensors) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)
* Huawei NPU support

// ... existing code ...

### Using Re-rankers models

`text-embeddings-inference` v0.4.0 added support for CamemBERT, RoBERTa and XLM-RoBERTa Sequence Classification models.
Re-rankers models are Sequence Classification cross-encoders models with a single class that scores the similarity
between a query and a text.

See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

```shell
model=BAAI/bge-reranker-large
revision=refs/pr/4
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.2 --model-id $model --revision $revision
```

And then you can rank the similarity between a query and a list of documents with:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "documents": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

返回格式示例:
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9997739
    },
    {
      "index": 1,
      "relevance_score": 0.3638598
    }
  ]
}
```



## Hardware Support

### Huawei NPU

此版本已添加对华为NPU的支持，使模型能够在华为NPU硬件上高效运行。通过利用华为NPU的计算能力，可以加速文本嵌入和重排序操作。

要在华为NPU上运行：

```shell
# 启动服务
model=BAAI/bge-reranker-large
text-embeddings-router --model-id $model --port 8080
```


## API 格式变更

### Rerank 接口适配

为了更好地与Dify和FastGPT等系统集成，我们更新了rerank接口的格式：

**原始接口格式**:
```bash
curl 127.0.0.1:12347/rerank \
-X POST \
-d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is a sub-filed of Machine Learning.", "Deep learning is a country."]}' \
-H 'Content-Type: application/json'
```

**现在的接口格式**:
```bash
curl 127.0.0.1:12347/rerank \
-X POST \
-d '{"query":"What is Deep Learning?", "documents": ["Deep Learning is a sub-filed of Machine Learning.", "Deep learning is a country."]}' \
-H 'Content-Type: application/json'
```

**返回格式**:
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9997739
    },
    {
      "index": 1,
      "relevance_score": 0.3638598
    }
  ]
}
```

这一变更使接口与Dify和FastGPT系统完全兼容，便于集成到现有的RAG工作流中。