# House Price Prediction — Linear Regression

Predicts a house's sale price from **square footage, number of bedrooms, and
number of bathrooms**, trained on the Ames Housing data (`train.csv`) from
your `data.zip`.

## Setup
```
pip install -r requirements.txt
```

## Expected folder structure
You can use either of these layouts:
```
your-project/
├── house_price_model.py
├── train.csv
├── test.csv
├── data_description.txt
└── sample_submission.csv
```
or
```
your-project/
├── house_price_model.py
└── data/
    ├── train.csv
    ├── test.csv
    ├── data_description.txt
    └── sample_submission.csv
```

## Run
```
python house_price_model.py
```

## What the script does
1. Loads `train.csv` (1,460 labeled houses)
2. Engineers `TotalBath = FullBath + 0.5 * HalfBath` (a "2.5 bath" house)
3. Drops 4 documented outliers — homes over 4,000 sq ft that sold far under
   market value (noted by the dataset's author as likely non-market sales)
4. Splits the data 80/20 into train/test
5. Fits `sklearn.linear_model.LinearRegression`
6. Reports MAE, RMSE, and R² on both splits
7. Saves two plots to `outputs/`: feature-vs-price scatter plots, and
   actual-vs-predicted + residuals
8. Exposes `predict_price(sqft, bedrooms, bathrooms)` for new inputs

## Results
| Split | MAE | RMSE | R² |
|-------|-----|------|-----|
| Train | $34,734 | $48,056 | 0.617 |
| Test  | $36,460 | $48,091 | 0.559 |

Fitted equation:
```
SalePrice ≈ 46,578 + 125·GrLivArea − 30,413·BedroomAbvGr + 18,859·TotalBath
```

## Why the bedroom coefficient is negative
This is not a bug. Holding square footage fixed, cramming in more bedrooms
means each room is smaller — buyers tend to pay less for that. It's a
confounding effect between `BedroomAbvGr` and `GrLivArea`, and worth being
able to explain if this gets reviewed.

## Known limitations
- R² ≈ 0.56 means these three features explain about 56% of price
  variance — reasonable, since the full dataset has 79 available predictors
  (models using all of them reach R² ≈ 0.9 on this data).
- The residual plot shows mild "funnel" spread at higher prices
  (heteroscedasticity) — a common violation of linear regression's constant-
  variance assumption in price data. Log-transforming `SalePrice` is the
  standard next step if you extend this.
