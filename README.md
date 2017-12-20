# Kaggle House Prices
---
The goal of this project is to show the code and development progress for the 
[Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition looking to predict housing prices in Ames, Iowa. This methodology 
uses regression to predict the wage and Machine Learning techniques to identifye
features and train and ideal model

## How to use
---
Everything is written in Python (version 3.4) with Jupyter Notebook. Users can easily install
both of these with [Anaconda](https://anaconda.org/anaconda/python). In addition to these, our
code relies on the following packages:
- [numpy](www.numpy.org/) to work efficiently with numbers
- [pandas](https://pandas.pydata.org/) to make data structuring and manipulation more manageable
- [ShuffleSplit](scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html) from sklearn.model_selection to mix our data before splitting into training/testing
- [r2_score](scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) from sklearn.metrics for our performance metric for each permutation of GridSearchCV
- [train_test_split](scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from sklearn.model_selection to split our data into training/testing sets
- [Imputer](scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) from sklearn.preprocessing to assign numeric values to missing values in our datasets
- [DecisionTreeRegressor](scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from sklearn.tree to build regressions from a decision tree
- [make_scorer](scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from sklearn.metrics to quickly make a performance metrics
- [GridSearchCV](scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) from sklearn.model_selection to iterate over multiple permutations of data
- [AdaBoostRegressor](scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) from sklearn.ensemble is an ensemble method we can apply to decision tree regressor
- [SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) from sklearn.linear_model is a stochastic gradient descent model
- [GradientBoostingRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) from sklearn.ensemble is a boosted gradient descent regression algorithm
- [pyplot](https://matplotlib.org/api/pyplot_api.html) from matplotlib to visualize results 

To sucessfully run the code, the user must set their working directory to the repository as it references a data
folder housing our data file `train.csv`. We'll then separate the target feature `SalePrice` from the other features
to train and explore different Decision Tree Regression models. If the working directory is correctly pointing to the
dat folder and referncing the `train.csv` file, then running all cells should yield the results of this analysis. The 
output are the training and testing scores of each model on the data given to us.
