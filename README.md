# Overview

This project addresses the need for robust and accurate tracking of cell nuclei in multichannel fluorescence microscopy videos with varying intensity levels. Our primary objective is to analyze changes in fluorescence intensity for each cell over time, leveraging a method that formulates tracking as an integer linear programming problem. The project uses fluorescence videos with four different channels, each representing a specific fluorophore located in the cell nuclei which are expressed in the G1 and S parts of the cell growth cycle and decay at different rates.
To track each cell, we propose an approach which segments each cell across each frame independently over each channel. We utilize an upcoming library called ultrack [1], which generates a hierarchy of segmentations by maximizing cell overlap between adjacent frames. It does so by using multiple segmentations hypothesis and maximizing the maximum total intersection over union as an integer linear problem (ILP). We use these independent segmentations as segmentation hypothesis which overcomes uncertainty in segmentation due to changing fluorescence intensities. We use StarDist [2] and Cellpose [3] due to the existence of out-of-box models trained on cell nuclei data.
Lastly, we implement a post-processing step that utilizes the unused intensity information by splitting tracks with jumps in relative intensity changes and prune short tracks. We provide code to build a host of visualizations to visualize and aggregate track data.

[1] J. Bragantini, M. Lange and L. Royer, "Large-Scale Multi-Hypotheses Cell Tracking Using Ultrametric Contours Maps," August 2023.

[2] U. Schmidt, M. Weigert, C. Broaddus and G. Myers, "Cell Detection with Star-Convex Polygons," in Lecture Notes in Computer Science, Springer International Publishing, 2018, p. 265–273.

[3] C. Stringer, T. Wang, M. Michaelos and M. Pachitariu, "Cellpose: a generalist algorithm for cellular segmentation," Nature Methods, vol. 18, p. 100–106, December 2020.

# Project Setup Guide

This README provides a step-by-step guide to setting up the project environment.

## Prerequisites

Before proceeding, ensure you have the following installed:

1. **Git**: [Git Download](https://www.git-scm.com/download/win)
2. **Miniconda**: [Miniconda Install Guide](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

## Installation Steps

### 1. Clone the Repository
Clone the project repository to your local machine.
```bash
git clone git@github.com:joris-gentinetta/DSL.git
```

### 2. Navigate to the Environments Folder
Change directory to the `envs` folder in the cloned repository.
```bash
cd DSL/envs
```

### 3. Change Conda Backend to Mamba
Switch the Conda backend to Mamba for efficient environment management.
```bash
source mamba.sh
```

### 4. Create the Environment
Set up the project environment using the provided script.
```bash
source create_env.sh
```

### 5. Create a Gurobi License
Obtain a "Named-User Academic" Gurobi License.
[Gurobi License Request](https://portal.gurobi.com/iam/licenses/request)

### 6. Set the Gurobi License
Activate your Gurobi license using the provided key.
```bash
grbgetkey <your key>
```

### 7. Navigate to the Project Folder
Return to the main project directory.
```bash
cd ..
```

## Running the Project

### Option 1: Run Everything Together
Execute the entire pipeline with a single command. Choices for configuration are `very_fast`, `fast`, `high_quality`.
```bash
source run_everything.sh demo.tif very_fast
```

### Option 2: Run Each Step Separately

#### Run the Segmentation Script
```bash
python segment/main.py --file demo.tif --config_id very_fast
```

#### Run the Tracking Script
```bash
python track/track.py --file demo.tif --config_id very_fast
```

#### Run the Postprocessing Script
```bash
python track/postprocess.py --file demo.tif --config_id very_fast
```

#### Display the Results
```bash
python display.py --file demo.tif --config_id very_fast
```

### Visualize Results
Utilize the `visualizations.ipynb` notebook for detailed visualization of the results.

