# Kaggle - Housing Prices Regression Analysis
## Set-Up for Kaggle Report

### Import Packages
```Python
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
```
### Import Data
```Python
data_train = pd.read_csv('data/train.csv')
```
### Splitting Features and Target Data
```Python
prices = data_train['SalePrice']
features = data_train.drop('SalePrice', axis=1)
```
### Scoring Function
```Python
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score
```
### Making Dummy Variable
Removing categorical data for regression analysis
```Python
features = pd.get_dummies(features, dummy_na = True)
```

### Removing NaN Values from DataFrame
```Python
fill_NaN = Imputer(missing_values='NaN', strategy='mean', axis=1)

# Create new Data rame with NaN values replaced with the column mean
features2 = pd.DataFrame(fill_NaN.fit_transform(features))
features2.columns = features.columns
features2.index = features.index
```

### Splitting into Training and Testing Sets
```Python
X_train, X_test, y_train, y_test = train_test_split(features2,
                                                    prices,
                                                    test_size=0.2,
                                                    random_state=0)
# Check for Null/NaN values
# print(X_train.isnull().values.any())
# print(y_train.isnull().values.any())
```

### Creating Cross-Validation Sets
```Python
cv_sets = ShuffleSplit(n_splits=10,
                       test_size=0.2,
                       random_state=0)
```

### Decision Tree Regression
```Python
tree = DecisionTreeRegressor(random_state=0)
params = {'max_depth': range(1, 11)}
scoring_fnc = make_scorer(performance_metric)
```

### GridSearch for Optimal Model
```Python
tree_grid = GridSearchCV(estimator=tree,
                    param_grid=params,
                    scoring=scoring_fnc,
                    cv=cv_sets)
tree_grid.fit(X_train, y_train)
```

### Making Predictions
```Python
tree_pred_train = tree_grid.predict(X_train)
tree_perf_train = performance_metric(y_train, tree_pred_train)
print("The r2-score on the training data was", round(tree_perf_train, 4))
tree_pred_test = tree_grid.predict(X_test)
tree_perf_test = performance_metric(y_test, tree_pred_test)
print("The r2-score on the testing data was", round(tree_perf_test, 4))
```


