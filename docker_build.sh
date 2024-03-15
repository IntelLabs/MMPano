#! /bin/bash
while [ "$1" != "" ];
do
   case $1 in
    -d | --device)
        DEVICE=$2
        shift
        ;;
    -h | --help )
         echo "Build the docker image for Multimodal Panorama Generation"
         echo "Usage: docker_build.sh [OPTIONS]"
         echo "OPTION includes:"
         echo "   -d | --device - Supported device [hpu]"
         exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Please do 'docker_build.sh -h'"
        exit
       ;;
  esac
  shift
done

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${CUR_DIR}/.env

DEVICE="${DEVICE:-hpu}"
DOCKERFILE=${CUR_DIR}/Dockerfile/Dockerfile-${DEVICE}

cmd="DOCKER_BUILDKIT=0 docker build . -f ${DOCKERFILE} -t ${IMAGE_NAME}_${DEVICE}:${IMAGE_TAG}"
echo $cmd
eval $cmd
