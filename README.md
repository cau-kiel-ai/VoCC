# VoCC Correlation Clustering

We introduce in this project the code to the paper **VoCC: Vortex Correlation Clustering based on masked Hough Transformation in Spatial Databases** together with its experiments.  

![](fig/overview.pdf)

## Vortex Correlation Clustering
Vortex Correlation Clustering is used to find vortices in two dimensional particles sets which have also a movement vector. For targeted radii the **VoCC** algorithm finds candidates with a Circle Hough Transformation in the two dimensional space and merges them to bigger clusters with our proposed **RDBSCAN** algorithm. This variant of DBSCAN does not need a specified epsilon and uses the radius of each vortex candidate. After identifying vortex candidates we map them back as labels onto the input data.

The Vortex Correlation Clustering algorithm is packed into an python package which can be directly installed with pip or after the anonymous phase we will publish this package on an public repository. The implementation is similar to the `sklearn` architecture and can be used accordingly.

## Experiments
We provide the code of our experiments in the `experiment/` folder in form of jupyter-notebooks. This notebooks are named according to the sections of the paper. Also the code for the visualization is provided to reproduce the figures within the paper.

## Reproduce 
For the reproducibility of the experiments we provide a Dockerfile in the `docker/` folder to build a container which is able to reproduce our results. This container launches an Jupyter server which can be exposed to the host system and reached by an browser to execute the code. To build the docker image naviagte to the project folder and build the image: 

`docker build -t vocc-jupyter -f docker/Dockerfile .`

After the image is created you can start the jupyter instance with the execution of the image. Again navigate to the project folder so that the src files can be mounted into the container to install the package:

`docker run --rm -p8888:8888 -v$(pwd):/home/jovyan/work/ vocc-jupyter`

The console will print an address to the jupyter lab as localhost address which can be opend in any browser or used in Visual Studio Code as an remote jupyter instance.