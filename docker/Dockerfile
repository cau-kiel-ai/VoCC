FROM jupyter/scipy-notebook:python-3.9.7

USER root

RUN apt update && \
    apt install -y build-essential

RUN conda install xarray=2023.6.0 cartopy=0.21.1 pandas=1.4.2 scikit-learn=1.3.0 

RUN pip3 install pyeddytracker==3.6.0 ipykernel==6.29.3 oceansdb==0.8.15 area==1.1.1

USER ${NB_UID}

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR ${HOME}/work

CMD pip3 install -e ${HOME}/work/;start-notebook.sh