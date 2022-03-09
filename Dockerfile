FROM python:3.9.10-slim-buster
WORKDIR /usr/src/app
RUN echo 'alias tst="python -m pytest arrays"' >> ~/.bashrc
RUN echo 'alias dtst="python -m pytest arrays --pdb"' >> ~/.bashrc
RUN echo 'alias jlab="jupyter lab --allow-root --ip 0.0.0.0 --no-browser"' >> ~/.bashrc
ENV PYTHONFAULTHANDLER=1
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
