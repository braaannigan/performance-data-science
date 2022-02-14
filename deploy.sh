#
#   Copyright (C) 2020 Liam Brannigan
#
#Target can be 'dev', 'test' or 'deploy' as per Dockerfile stages
TARGET=$1

DOCKER_BUILDKIT=1 docker build -t perf-data-science .

if [ "${TARGET}" == "dev" ]; then
docker run  -it --rm  -v $(pwd):/usr/src/app perf-data-science:latest /bin/bash
fi

if [ "${TARGET}" == "app" ]; then
docker run -it --rm -p 8501:8501  -v $(pwd):/usr/src/app perf-data-science:latest /bin/bash
fi

if [ "${TARGET}" == "lab" ]; then
docker run -it --rm -p 8888:8888  -v $(pwd):/usr/src/app perf-data-science:latest /bin/bash
fi
