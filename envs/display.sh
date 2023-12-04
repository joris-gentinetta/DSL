
conda env remove -p display -y
conda create -n display python=3.10 -y
conda activate display || exit
conda install -c conda-forge napari pyqt
conda install matplotlib