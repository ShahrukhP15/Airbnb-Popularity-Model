# Airbnb Popularity Prediction

A machine learning model that predicts how popular a New York City Airbnb listing is, using reviews per month as a proxy for popularity.

**Result:** a tuned LightGBM model predicts reviews per month with **31% lower mean absolute error** than a baseline that always predicts the average (test MAE 0.86 vs 1.26 reviews per month).

## Highlights
- **Leakage-safe pipeline:** a custom scikit-learn transformer learns neighbourhood features inside each cross-validation fold, so validation and test data never leak into training.
- **Model comparison:** Ridge regression, decision tree, Random Forest and LightGBM, compared with 5-fold cross-validation.
- **Tuning and selection:** hyperparameter search with RandomizedSearchCV, and feature selection tested with RFECV.
- **Interpretability:** SHAP analysis explains which features drive predictions, overall and for individual listings.
- **Reproducible:** the data downloads automatically, a fixed seed is used throughout, and fitted models, search results and package versions are saved to `artifacts/`.

## Data
[New York City Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) from Kaggle: 48,895 listings with location, room type, price, minimum nights, availability, host and review information.

**Target:** `reviews_per_month`. 10,052 listings with no review rate were excluded rather than filled in, leaving 38,843 listings, split 70/30 into training (27,190) and test (11,653) sets.

## Approach
1. **Explore** distributions, outliers, and how review rates vary by borough and room type
2. **Engineer features:**
   - `minimum_stay_price`: price × minimum nights, the advertised cost of the shortest allowed stay
   - `neighbourhood_popularity`: total reviews in each neighbourhood, learned only from the training fold
3. **Preprocess** inside one pipeline: median imputation and scaling for numeric features, one-hot encoding for categorical features. Listing names, host details and review counts are not used as direct inputs.
4. **Compare models** against a dummy baseline with 5-fold cross-validation
5. **Tune** hyperparameters with RandomizedSearchCV and test RFECV feature selection
6. **Interpret** the final model with SHAP
7. **Evaluate** once on the held-out test set

## Results

Best mean cross-validation R² after tuning:

| Model | CV R² |
|---|---|
| Dummy baseline | 0.00 |
| Ridge regression | 0.10 |
| Decision tree | 0.31 |
| Random Forest | 0.39 |
| **LightGBM** | **0.40** |

LightGBM scored best and trained about 25 times faster than Random Forest. RFECV feature selection did not improve it (0.39 vs 0.40), so all features were kept.

Final LightGBM (301 trees, max depth 13, learning rate 0.1) on the test set:

| Metric | LightGBM | Baseline | Change |
|---|---|---|---|
| MAE (reviews/month) | 0.86 | 1.26 | 31% lower |
| RMSE (reviews/month) | 1.40 | 1.77 | 21% lower |
| R² | 0.38 | 0.00 | |

### What drives the predictions
The features with the largest average SHAP contributions:
1. **Availability** (days per year the listing is open)
2. **Minimum nights** required per booking
3. **Minimum stay price**
4. **Room type**, especially entire homes
5. **Location** (longitude)

These describe how the model makes predictions, not causal effects.

## Limitations
- The model explains about 38% of the variation in review rates. Much of what makes a listing popular, like photos, ratings and descriptions, is not in this dataset.
- Reviews per month is only a proxy for popularity. Occupancy or booking data would be a better target.
- The data is a single 2019 snapshot with a random split, so results don't show how the model would perform on future or brand-new listings.

## How to run
You need Python 3.9 or newer. Run these commands in a terminal (Terminal on macOS or Linux, Command Prompt or PowerShell on Windows).

1. Clone the repository and move into it:

       git clone https://github.com/ShahrukhP15/Airbnb-Popularity-Model.git
       cd Airbnb-Popularity-Model

2. (Optional) Create and activate a virtual environment, so the packages don't affect your other projects:

       python -m venv venv
       source venv/bin/activate        # macOS/Linux
       venv\Scripts\activate           # Windows

3. Install the dependencies:

       pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn joblib kagglehub jupyter

4. Launch Jupyter and open the notebook:

       jupyter notebook airbnb-model.ipynb

5. In Jupyter, choose **Run > Run All Cells**. The dataset downloads automatically from Kaggle the first time you run it.

`model_utils.py` must stay in the same folder as the notebook. Fitted models and results are saved to `artifacts/`, which is not tracked in git.

To browse the results without running anything, view the notebook on GitHub or open `airbnb-model.html` in a browser.


## Project structure

    ├── airbnb-model.ipynb    full analysis notebook
    ├── airbnb-model.html     notebook exported with outputs
    └── model_utils.py        leakage-safe feature transformer used by the pipeline

## Author
Shahrukh Islam Prithibi.
