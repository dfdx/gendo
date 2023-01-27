FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel as build-base

RUN apt update
# RUN apt install -y python3 python3-pip

# move installation of the most heavy dependencies here to speed up builds
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install pytorch-lightning==1.8.3
RUN pip3 install torchmetrics==0.11.0
RUN pip3 install transformers==4.24.0

# the rest of dependencies will be installed as usual from setup.py
COPY ./ /app
WORKDIR /app
RUN pip3 install .

# pulsar
RUN pip3 install yandex-pulsar-1.0.1.tar.gz

CMD ["echo", "Don't talk to me about life..."]


FROM build-base as build-dev

RUN mkdir -p ~/.cache/huggingface
RUN pip3 install ipython

FROM build-dev as build-dev-jupyter

RUN pip3 install pandas scikit-learn statsmodels jupyterlab ipywidgets matplotlib seaborn plotly
