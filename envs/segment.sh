conda env remove -n segment -y
conda env create -f segment_env.yml conda activate segment
conda activate segment || exit

