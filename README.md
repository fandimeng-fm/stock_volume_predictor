# Auto-Regressive Methods in Predictive Volume Modeling

### Description:
This is the codebase for the "Trading Volume Prediction of Instruments with Observable and Hidden Structure" project for 
the University of Chicago's Winter 2022 and Spring 2022 Data Science Clinic. The objective of this project was to build an online model for forecasting instrument volume smiles.

##### This codebase consists of:
 - Implementation of selected models outlined in "Quintet Volume Projection" (Markov et al., 2019) as a baseline comparison model
 - Scripts for improved intraday volume and daily total volume prediction models 
 - Analysis, comparisons and visualizations of model performance referenced in the accompanying paper
 - Analysis of consistent intraday "jumps" in volume found at regular 5 minute intervals

### Dependencies:
This program was written in Python 3.7. The full list of library requirements can be found in `requirements.txt` . 

### Execution:
To test `improved_intraday.py` and `improved_daily.py` on the sample Russell 3000 data included in this repository,
run `python3 <filename>` from the current directory.

An example for running the baseline model can be found in `baseline_sample.ipynb`. 

### File Directory:

The folders found in the current directory are:
- `baseline`: files for our baseline model
- `data`: datasets used in our model analysis as well as results of our model evaluation
- `jump_analysis`: Python notebooks that outline our research and observations into intraday volume jumps at regular intervals
- `notebooks`: Python notebooks containing the tables and figures included in our final paper, including other tested models

#### Proposed Models:

The proposed daily and intraday volume prediction models referenced in our paper can be found in `improved_daily.py` and
`improved_intraday.py`. The results of our model evaluation can be found in `data/daily_results`, 
`data/robust_arma_results`, and `data/volatility_results`. Python notebooks containing our analysis and visualization of the model results
can be found in the `notebooks` folder. 

#### Other Tested Models: 

We tested a number of models during our research and development process that underperformed the baseline or underperformed the final models we proposed. 
For reference, notebooks containing our analysis on those models and visualiztions used in our paper have been included in the `notebooks` folder. 
For posterity, all other notebooks utilized in our research over the course of the project have been included in the `notebooks/deprecated` folder, including additional incomplete model ideas. 
 
### Authors and Acknowledgments:

Team Members: Yanhao Dong, Ben Goldman, Xiaotian Li, Fandi Meng, Sushan Zhao

We wish to thank DRW Holdings and the University of Chicago’s Data Science Institute for providing us the opportunity to 
work on this project. We extend our sincerest gratitude to our mentors from the University of Chicago: 
Phillip Lynch, David Uminsky, Dan Nicolae, Cole Bryant and Nick Gondek for their guidance and support. 
