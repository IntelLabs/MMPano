#! /bin/bash
model=meta-llama/Meta-Llama-3-8B-Instruct
CONTAINER_NAME=tgi-gaudi
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
num_shard=2
sharded=true
max_input_length=2048
max_total_tokens=4096

# Usage: text-generation-launcher <
#     --model-id <MODEL_ID>|
#     --revision <REVISION>|
#     --validation-workers <VALIDATION_WORKERS>|
#     --sharded <SHARDED>|
#     --num-shard <NUM_SHARD>|
#     --quantize <QUANTIZE>|
#     --speculate <SPECULATE>|
#     --dtype <DTYPE>|
#     --trust-remote-code|
#     --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>|
#     --max-best-of <MAX_BEST_OF>|
#     --max-stop-sequences <MAX_STOP_SEQUENCES>|
#     --max-top-n-tokens <MAX_TOP_N_TOKENS>|
#     --max-input-tokens <MAX_INPUT_TOKENS>|
#     --max-input-length <MAX_INPUT_LENGTH>|
#     --max-total-tokens <MAX_TOTAL_TOKENS>|
#     --waiting-served-ratio <WAITING_SERVED_RATIO>|
#     --max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>|
#     --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>|
#     --max-waiting-tokens <MAX_WAITING_TOKENS>|
#     --max-batch-size <MAX_BATCH_SIZE>|
#     --cuda-graphs <CUDA_GRAPHS>|
#     --hostname <HOSTNAME>|
#     --port <PORT>|
#     --shard-uds-path <SHARD_UDS_PATH>|
#     --master-addr <MASTER_ADDR>|
#     --master-port <MASTER_PORT>|
#     --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>|
#     --weights-cache-override <WEIGHTS_CACHE_OVERRIDE>|
#     --disable-custom-kernels|
#     --cuda-memory-fraction <CUDA_MEMORY_FRACTION>|
#     --rope-scaling <ROPE_SCALING>|
#     --rope-factor <ROPE_FACTOR>|
#     --json-output|
#     --otlp-endpoint <OTLP_ENDPOINT>|
#     --cors-allow-origin <CORS_ALLOW_ORIGIN>|
#     --watermark-gamma <WATERMARK_GAMMA>|
#     --watermark-delta <WATERMARK_DELTA>|
#     --ngrok|
#     --ngrok-authtoken <NGROK_AUTHTOKEN>|
#     --ngrok-edge <NGROK_EDGE>|
#     --tokenizer-config-path <TOKENIZER_CONFIG_PATH>|
#     --disable-grammar-support
# 

# -e HUGGING_FACE_HUB_TOKEN=<YOUR_HF_TOKEN> \
docker run \
    -p 8080:80 \
    -v $volume:/data \
    --runtime=habana \
    -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice \
    --ipc=host \
    --name=${CONTAINER_NAME} \
    ghcr.io/huggingface/tgi-gaudi:2.0.0 \
    --model-id $model --sharded $sharded --num-shard $num_shard --max-input-length $max_input_length --max-total-tokens $max_total_tokens
