#! /bin/bash
# HABANA_VISIBLE_DEVICES, HABANA_VISIBLE_MODULES
# 0, 2
# 1, 6
# 2, 0
# 3, 7
# 4, 1
# 5, 4
# 6, 3
# 7, 5

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${CUR_DIR}/.env

DEVICE_IDX=0
MODULES_IDX=2
IMAGE_NAME=${IMAGE_NAME}_hpu:${IMAGE_TAG}
CONTAINER_NAME=${CONTAINER_NAME}_hpu

OUTPUT_DIR_LOCAL=./exp
OUTPUT_DIR_CONTAINER=/app/outputs
docker run -it \
        --expose=${API_PORT} \
        --expose=${WEBAPP_PORT} \
        -v ${OUTPUT_DIR_LOCAL}:${OUTPUT_DIR_CONTAINER} \
        --env=DEVICE=hpu \
        --env=HABANA_VISIBLE_DEVICES=all \
        --env=OMPI_MCA_btl_vader_single_copy_mechanism=none \
        --cap-add=sys_nice \
        --network=host \
        --restart=no \
        --runtime=habana \
	--shm-size=64g \
        --name ${CONTAINER_NAME} \
        -t ${IMAGE_NAME} 
