# Bike Sharing Demand Prediction with AutoGluon
A machine learning project that predicts bike sharing demand using AutoGluon's automated ML capabilities.
Project Overview
This project uses AutoGluon to predict bike sharing demand based on weather conditions, time, and seasonal factors. The model achieved a significant 59% improvement in performance through strategic feature engineering.


Key Results

Best Model: Feature-engineered AutoGluon ensemble

Performance: 0.53836 RMSE (59% improvement over baseline)

Top Features: Hour of day, weather conditions, temperature


Key Features Added

Hour of day: Captures rush hour patterns and daily cycles

Categorical encoding: Proper handling of season and weather as categories

Temporal patterns: Leverages time-based demand variations


Technologies Used

AutoGluon: Automated machine learning framework

Pandas: Data manipulation and analysis

Python: Core programming language

Kaggle: Competition platform and dataset


Key Insights

Feature engineering > Hyperparameter tuning: Adding meaningful features had 20x more impact than HPO

Domain knowledge matters: Understanding bike sharing patterns (rush hours, weather) was crucial

AutoGluon effectiveness: Ensemble methods worked well out-of-the-box


Files Structure
├── project-template.ipynb    # Main analysis notebook

├── img/

│   ├── model_train_score.png       # Training performance

│   └── model_test_score.png        # Kaggle scores

└── README.md                       

Quick Start

Load the bike sharing dataset

Run exploratory data analysis

Engineer temporal features (hour extraction)

Train AutoGluon model with best_quality preset

Submit predictions to Kaggle


Author
HAJER TALBI 

This project demonstrates the power of feature engineering in ML and AutoGluon's effectiveness for tabular data prediction tasks.
