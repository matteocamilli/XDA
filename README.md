# Explanation-Driven Self-Adaptations (XDA) Replication Package

## Install Requirements
```pip install -r requirements.txt```

### For macOS and Linux Users
```chmod +x MDP_Dataset_builder/evaluate_adaptations.sh```

## Generate Dataset (Optional)
There are 3 directory in MDP_Dataset_Builder with the name of test instances (RescueRobot, UAV_v02 and AutonomousDriving_v1)

Move the config.py file of the selected directory in the MDP_Dataset_Builder directory

Inside MDP_Dataset_builder/run.sh and MDP_Dataset_builder/run.bat:
* MAX_SAMPLES: number of samples to generate
* TOTAL_THREADS: number of threads to use for the generation

### For macOS and Linux Users
```
chmod +x MDP_Dataset_builder/run.sh
./run.sh
```

### For Windows Users
```.\run.bat```

## Run Adaptation Tests
Firstly move the config.py file in MDP_Dataset_Builder directory (as written in Generate Dataset)

Inside main/main.py:

* line 68: you can specify the path to your dataset
* lines 69-70-71-72: you can select the features used (first are for uav, second for rescue robot and third for autonomous driving)
* lines 82-83-84-85: you can specify the list of requirements to consider (first are for autonomous driving, second for rescue robot and last for uav)
* line 87: you can specify the size of the neighborhood
* line 88: you can specify the number of starting solutions to consider
* line 91: you can specify the target success probabilities for each requirement
* line 181: you can specify the number of tests to do

```python main/main.py```

## Generate Plots
Go to main/resultAnalyzer.py

* line 301: you can specify the path to your results
* lines 303-304-305: you can specify the features used (first for uav, second for rescue robot and the last for autonomous driving)
* line 307-308-309-310: you can specify which requirements are used (first for uav, second for rescue robot and the last for autonomous driving)

```python main/resultAnalyzer.py```

## Profiling
Inside main/offlineProfiler.py

* line 77: specify path to dataset
* lines 78-79-80-81: you can specify the features used (first for uav, second for rescue robot and the last for autonomous driving)
* lines 86-87-88-89: you can specify the requirements used (first for uav, second for rescue robot and the last autonomous driving)
* lines 120-121: to collect data (time and memory) on PDP and SPDP construction
* lines from 134 to 188: to collect data (time and memory) on SHAP and permutation feature importance

```python main/offlineProfiler.py```

