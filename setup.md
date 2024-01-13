install git:
```
https://www.git-scm.com/download/win
```

install miniconda:
```
https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
```

clone the repo:
```
git clone git@github.com:joris-gentinetta/DSL.git
```

go to the envs folder:
```
cd DSL/envs
```

change the conda backend to mamba:
```
source mamba.sh
```

create the environment:
```
source create_env.sh
```

create a "Named-User Academic" Gurobi License:
```
https://portal.gurobi.com/iam/licenses/request
```

set the Gurobi license
```
grbgetkey <your key>
```

go to the project folder
```
cd ..
```

run the segmentation script:
```
python segment/main.py --file demo.tif --config_id high_quality
```

run the tracking script:
```
python track/track.py --file demo.tif --config_id high_quality
```



