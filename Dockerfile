FROM nvcr.io/nvidia/pytorch:23.07-py3
WORKDIR /home/${USER}
#ENV PATH="${PATH}:/home/${USER}/.local/bin"

LABEL maintainer="sridhark@email.unc.edu"


ARG USER_ID
ARG GROUP_ID
ARG UNAME
RUN groupadd -g $GROUP_ID -o $UNAME
RUN useradd -m -u $USER_ID -g $GROUP_ID -o -s /bin/bash $UNAME
USER root
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update --allow-unauthenticated
RUN apt-get install -y git
WORKDIR /opt

# Repo requirements
RUN python -m pip install -U pip
COPY repo_requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
# RUN python -m pip install -U jupyterlab jupyterlab_widgets ipywidgets
# ENTRYPOINT ["jupyter", "lab", "--port=8888", "--notebook-dir=/", "--no-browser", "--ip=0.0.0.0"]
