# Low Latency Trajectory Predictions For Interaction Aware Highway Driving

by Cyrus Anderson at [UM FCAV](https://fcav.engin.umich.edu/)

### Introduction

The paper proposes a probabilistic model to predict trajectories of vehicles in critical highway merging scenarios, where the number of observations may be limited.
For more details, check out the published paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/9140336) or the [arxiv](https://arxiv.org/abs/1909.05227) version.


### Dependencies

- NumPy
- SciPy
- PICOS
- CVXOPT
- pandas
- matplotlib


The proposed method's semidefinite program is formulated and solved with the PICOS interface for conic optimization. The solver used is CVXOPT.
The other dependencies are used for loading data and plotting.

Here is an example of creating an environment named `py37sdp` with Anaconda having the dependencies:

```
conda create -n py37sdp python=3.7 numpy scipy pandas matplotlib pip 
pip install cvxopt
conda install -c picos picos
```

### Datasets


The NGSIM dataset is used to evaluate the method, whose root folder should be set in `utils.py`.
The default setup uses `datasets` as a symbolic link:
```
baselines/
datasets/
|__ngsim/
    |__i-80/
    |__us-101/
```
where the `ngsim` folder contains the NGSIM dataset ([dataset portal](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
and [homepage](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)).


(Note: There may be small errors when first loading the data due to
small inconsistencies between the folder name formats/column names
of the US-101 and I-80 datasets - manually changing them can solve this.)

### Predicting

Running `display_driver.py` will evaluate each baseline on ramp merge scenarios from NGSIM.

### Citation

If you find this paper helpful, please consider citing:
```
@ARTICLE{anderson2020lowlatency,
  author={C. {Anderson} and R. {Vasudevan} and M. {Johnson-Roberson}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Low Latency Trajectory Predictions for Interaction Aware Highway Driving}, 
  year={2020},
  volume={5},
  number={4},
  pages={5456-5463},
}
```
