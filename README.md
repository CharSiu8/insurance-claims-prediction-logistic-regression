# Insurance Claims Prediction with Logistic Regression

A logistic regression project comparing manual implementation with sklearn industry-standard workflow to predict whether car insurance customers will make a claim.

## Project Overview

This project demonstrates two approaches to logistic regression modeling:

1. **Simple Solution**: Blindly loops through features using statsmodels' logit function, then manually calculates accuracy using confusion matrix values (TP, TN, FP, FN).

2. **Analytical Workflow Solution**: Industry-standard approach with proper pre-processing, visualization, correlation heatmapping to pre-screen features, and sklearn validation.

## Business Context

Insurance companies invest significant time and money into optimizing pricing and estimating claim likelihood. This project was built for "On the Road" car insurance to identify the **single best feature** for predicting whether a customer will make a claim during their policy period.

The goal was to find a simple, production-ready model using just one feature, measured by accuracy.

## Results

**Best Predictor: `driving_experience`**  
**Accuracy: 77.71%**

Both the manual implementation and sklearn validation produced identical results, confirming `driving_experience` as the strongest single predictor of insurance claims.

## Dataset

The dataset contains **10,000 records** with **18 columns** of customer information:

| Column | Description |
|--------|-------------|
| `id` | Unique client identifier |
| `age` | Client's age group (0: 16-25, 1: 26-39, 2: 40-64, 3: 65+) |
| `gender` | Client's gender (0: Female, 1: Male) |
| `driving_experience` | Years driving (0: 0-9y, 1: 10-19y, 2: 20-29y, 3: 30+) |
| `education` | Education level (0: None, 1: High school, 2: University) |
| `income` | Income level (0: Poverty, 1: Working class, 2: Middle class, 3: Upper class) |
| `credit_score` | Credit score (0-1 scale) |
| `vehicle_ownership` | Owns vehicle (0: No/financing, 1: Yes) |
| `vehicle_year` | Registration year (0: Before 2015, 1: 2015 or later) |
| `married` | Marital status (0: Not married, 1: Married) |
| `children` | Number of children |
| `postal_code` | Client's postal code |
| `annual_mileage` | Miles driven per year |
| `vehicle_type` | Car type (0: Sedan, 1: Sports car) |
| `speeding_violations` | Number of speeding violations |
| `duis` | Number of DUI incidents |
| `past_accidents` | Number of previous accidents |
| `outcome` | Made a claim (0: No, 1: Yes) - **Target variable** |

**Download the data**: [Google Drive Link](https://drive.google.com/file/d/1SiDNh9YFIiC4NjekT9R1vVfOyAdIHiFX/view?usp=sharing)

## Methodology

### Data Preprocessing

1. **Missing Value Handling**: Filled `credit_score` (982 missing) and `annual_mileage` (957 missing) with mean values
2. **Categorical Encoding**: Converted object columns to ordinal numeric values:
   - `driving_experience`: 0-9y → 0, 10-19y → 1, 20-29y → 2, 30y+ → 3
   - `education`: high school → 0, university → 1, none → 3
   - `income`: poverty → 0, working class → 1, middle class → 2, upper class → 3
   - `vehicle_year`: before 2015 → 0, after 2015 → 1
   - `vehicle_type`: sedan → 0, sports car → 1

### Modeling Approach

**Simple Solution:**
- Loop through each feature column
- Fit logistic regression using `statsmodels.formula.api.logit`
- Extract confusion matrix via `model.pred_table()`
- Calculate accuracy: `(TP + TN) / (TP + TN + FP + FN)`

**Analytical Solution:**
- Correlation heatmap to visualize feature relationships with outcome
- Validate with `sklearn.metrics.accuracy_score`
- Compare results across all features

## Tech Stack

- Python 3.x
- pandas
- numpy
- statsmodels
- scikit-learn
- seaborn
- matplotlib

## Installation

```bash
pip install pandas numpy statsmodels scikit-learn seaborn matplotlib
```

## Usage

```python
import pandas as pd
from statsmodels.formula.api import logit

# Load data
car_insurance = pd.read_csv('car_insurance.csv')

# Preprocess and run models
# See notebook for full implementation
```

## Related Projects

Check out the [AI Data Auditor Model](https://github.com/CharSiu8/data_auditor.git) which uses this same data as test data.

## License

This project is open source and available for educational purposes.
