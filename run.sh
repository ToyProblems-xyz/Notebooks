
DOCKER_VOLUMES="
--volume="./src/:/app/src/" \
"

docker run -p 8888:8888 -i -t ${DOCKER_VOLUMES} tpxyz-notebooks:latest
