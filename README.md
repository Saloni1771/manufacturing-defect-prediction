# Manufacturing Defect Prediction — SECOM Sensor Data

Predicting failures on a semiconductor production line using sensor data from the UCI SECOM dataset.

I wanted to work on something closer to the manufacturing and reporting work I deal with rather than another generic tabular ML dataset.

## What I worked on

The dataset has 1,567 production units and 590 sensor readings for each unit. Only about 6.6% of the units are labelled as failures, so the classes are quite imbalanced.

The main goal was to see whether the sensor readings could be used to identify units that were likely to fail.

One thing that stood out during the project was that the model performance looked better from ROC-AUC than it did when I actually used the default 0.5 classification threshold. The Random Forest had the best ROC-AUC (0.835), but at 0.5 it didn't predict a single failure.

That made threshold selection more interesting than I initially expected.

## Data cleaning

There were a lot of sensor columns that weren't particularly useful for modelling.

I removed:

* ~28 sensors with a large amount of missing data
* ~265 sensors with almost no variation
* ~102 sensors that were highly redundant with other sensors

This reduced the dataset from 590 sensor columns to 195.

I also looked at the class distribution and whether the failure rate changed across the production sequence.

## Models

I compared three models:

* Logistic Regression
* Random Forest
* Gradient Boosting

Because the failure class was relatively small, I used class weighting rather than oversampling the minority class.

The initial results were:

| Model               | ROC-AUC | Recall @ 0.5 | Precision @ 0.5 |
| ------------------- | ------: | -----------: | --------------: |
| Random Forest       |   0.835 |        0.000 |           0.000 |
| Gradient Boosting   |   0.787 |        0.077 |           0.222 |
| Logistic Regression |   0.709 |        0.385 |           0.167 |

The Random Forest had the highest ROC-AUC, but that number alone was not enough to decide whether the model would actually be useful.

## Looking at the threshold

With only around 6.6% failures, using 0.5 as the cutoff meant that the Random Forest classified every test example as non-failure.

That obviously isn't useful if the objective is to catch defective units.

I therefore looked at what happened when the classification threshold was lowered. Using a threshold of around 0.11 gave recall of about 96%.

The trade-off was a large increase in false positives, meaning more units would need to be inspected.

For this project I used an approximate 50:1 cost assumption for a missed defect versus an unnecessary inspection. It's not a real fab cost model, so I wouldn't treat the exact 0.11 threshold as production-ready. The point was to show how the preferred threshold changes when missing a defect is much more expensive than doing an extra inspection.

## What I took from the project

The biggest thing I learned wasn't really that Random Forest performed better than the other two models.

It was that **the classification threshold matters a lot when the target is highly imbalanced**.

A model can have a reasonable ROC-AUC and still be practically useless if the threshold isn't appropriate for the problem.

The other part that took more work than I expected was the sensor cleaning. More than half of the original sensor columns were removed because of missingness, very low variation, or redundancy.

If I continued this project, I'd compare the class-weighted approach with SMOTE and other resampling methods. I'd also want to look more closely at the most important sensors and, ideally, validate whether the patterns identified by the model make sense from an actual semiconductor process perspective.

## Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Project structure

```text
├── data/                # SECOM dataset files
├── notebook/            # Full analysis notebook
├── outputs/             # Charts and model comparison results
└── analysis.py          # Analysis as a Python script
```

Dataset: UCI SECOM
https://archive.ics.uci.edu/dataset/179/secom
