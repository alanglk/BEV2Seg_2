#!/bin/bash

# Version parsing
if [ -z "$1" ]; then
  echo "[SCRIPT]    Error: A docker image version must be provided: ./build_docker_image.sh v0.1"
  exit 1
fi

# Docker image name:tag
VERSION=$1
IMAGE_NAME="beg2seg_2"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"

# Dockerfile path: ./docker
DOCKERFILE_DIR="docker"

# Construir la imagen Docker
echo "[SCRIPT]    Building Docker image ${IMAGE_TAG}..."
docker build -t "${IMAGE_TAG}" -f "${DOCKERFILE_DIR}/Dockerfile" .
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t "${IMAGE_TAG}" -f "${DOCKERFILE_DIR}/Dockerfile" .

if [ $? -ne 0 ]; then
  echo "[SCRIPT]    Error: Unnexpected error building docker image."
  exit 1
fi

# Save Docker image as a .tar file for singularity conversion
TAR_FILE="${DOCKERFILE_DIR}/${IMAGE_NAME}_${VERSION}.tar"
echo "[SCRIPT]    Saving Docker image as ${TAR_FILE}..."
docker save -o "${TAR_FILE}" "${IMAGE_TAG}"

if [ $? -ne 0 ]; then
  echo "[SCRIPT]    Error: couldn't save image as .tar."
  exit 1
fi