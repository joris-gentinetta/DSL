conda env remove -n segment_stardist -y
conda create -n segment_stardist python=3.10 -y
conda activate segment_stardist || exit
pip install tensorflow
pip install stardist
