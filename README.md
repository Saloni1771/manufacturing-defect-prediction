# Manufacturing Defect Prediction — SECOM Sensor Data

Predicting failures on a semiconductor production line using sensor data (UCI SECOM dataset). Wanted to work with something closer to the manufacturing/reporting domain I deal with at my day job, instead of another generic tabular dataset.

## What this is

1,567 units, 590 sensor readings each, ~6.6% failure rate. So — real imbalance, not a clean 50/50 split. Task: catch failing units from the sensor data before they ship.

Best model came out to Random Forest (ROC-AUC 0.835), but honestly the more useful finding wasn't the model — it was what happened when I actually tried to use it at the default threshold. It caught *zero* failures. Turns out with this much imbalance, 0.5 is a bad cutoff by default, and you have to go find the threshold that actually makes sense for the cost of missing a defect vs. flagging a false alarm. More on that below.

## What's in here

- EDA on the sensors — missingness, class balance, whether failure rate drifted over time
- Cleaning: dropped ~28 sensors that were mostly missing, ~265 that were basically constant (dead sensors, I'm guessing), and another ~102 that were redundant with something else. Went from 590 columns down to 195.
- Three models — logistic regression, random forest, gradient boosting — using class weighting instead of the more common SMOTE oversampling (didn't have the library available, class_weight='balanced' works fine and is arguably more transparent anyway)
- A cost-based threshold analysis, since accuracy/default-threshold numbers are close to meaningless on a dataset this imbalanced

## Results

| Model | ROC-AUC | Recall @ 0.5 | Precision @ 0.5 |
|---|---|---|---|
| Random Forest | 0.835 | 0.000 | 0.000 |
| Gradient Boosting | 0.787 | 0.077 | 0.222 |
| Logistic Regression | 0.709 | 0.385 | 0.167 |

At the default threshold the "best" model is useless — it never predicts a failure. Moved the threshold down to ~0.11 (based on assuming a missed defect costs roughly 50x an unnecessary inspection — rough estimate, but directionally makes sense for a fab) and recall jumps to 96%. Precision drops a lot in exchange, which is the actual trade-off you're making, not a flaw.

## Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Structure

```
├── data/                # secom.data, secom_labels.data, secom.names
├── notebook/             # full analysis notebook
├── outputs/               # charts + model comparison csv
└── analysis.py             # same analysis as a script
```

## Takeaways

Model choice mattered less here than I expected going in — all three found some signal. What actually mattered was realizing the default threshold was silently assuming false positives and false negatives cost the same, which isn't true in a QA context. Also spent a good chunk of time just on cleaning — dead/constant sensors and redundant columns made up more than half of the original 590.

Next time I'd try proper resampling (SMOTE) for comparison, and ideally get someone with actual fab process knowledge to sanity-check whether the top sensors the model picked up on make physical sense.

Dataset: [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)
