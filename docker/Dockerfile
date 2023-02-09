FROM jupyter/scipy-notebook:python-3.9.7

USER root

RUN apt update && \
    apt install -y build-essential

RUN conda install geoviews cartopy cartopy_offlinedata xarray

RUN pip3 install pyEddyTracker==3.6.0

USER ${NB_UID}

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR ${HOME}/work

CMD pip3 install -e ${HOME}/work/;start-notebook.sh