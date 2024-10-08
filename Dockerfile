FROM docker.io/dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
RUN apt update && apt install -y curl gpg software-properties-common

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install \
    boto3 \
    botocore


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/amazon-bedrock-adapters:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/amazon-bedrock-adapters:0.0.1

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/amazon-bedrock-adapters:0.0.1 bash