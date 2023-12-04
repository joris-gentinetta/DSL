
conda env remove -n track -y
conda create -n track python=3.10 -y
conda activate track || exit
cd ../../../.. || exit
pip install ultrack
conda install pytorch cpuonly -c pytorch -y
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c gurobi gurobi #todo license
#grbgetkey 9ab029a7-3527-434e-9d3b-64ccde1aedba