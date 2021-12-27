FROM gpuci/miniforge-cuda:11.0-runtime-ubuntu18.04

COPY . /app
RUN conda install -y mamba
RUN mamba env create -f /app/environment.yaml
RUN conda clean --all --yes

WORKDIR /app
ENTRYPOINT ["/opt/conda/envs/decode_dev/bin/python", "-m", "decode.neuralfitter.train.train"]
