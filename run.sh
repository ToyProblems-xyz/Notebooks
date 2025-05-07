
DOCKER_VOLUMES="
--volume="./src/:/app/src/" \
"

docker run -i -t ${DOCKER_VOLUMES} tpxyz-template:latest /bin/bash
