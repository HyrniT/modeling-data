# MODELING DATA

## Intro

* Learning Objectives:

  * Become familiar with pandas for handling small datasets
  * Use the tf.Estimator and Feature Column API to experiment with feature transformations
  * Use visualizations and run experiments to understand the value of feature transformations

## Usage
### Pandas, a helpful data analysis library for in-memory dataset

* We use a package called [Pandas](http://pandas.pydata.org/) for reading in our data, exploring our data and doing some basic processing. It is really helpful for datasets that fit in memory! And it has some nice integrations, as you will see.

* First we set up some options to control how items are displayed and the maximum number of rows to show when displaying a table. Feel free to change this setup to whatever you'd like.

```python
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 15
```

### Load the dataset with pandas

* The car data set we will be using in this lab is provided as a comma separated file without a header row. In order for each column to have a meaningful header name we must provide it. We get the information about the columns from the [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile).

* We will use the features of the car, to try to predict its price.

```python
car_data = pd.read_csv('https://storage.googleapis.com/mledu-datasets/cars_data.csv',
                        sep=',', names=feature_names, header=None, encoding='latin-1')

# Then randomize the datat
car_data = car_data.reindex(np.random.permutation(car_data.index))
```
### Use pandas to explore and prepare the data

```python
car_data[4:7]
```

```python
# Run to inspect numeric features.
car_data[numeric_feature_names]
```

```python
# Run to inspect categorical features.
car_data[categorical_feature_names]
```

### Training data

* Modify the model provided to achieve the lowest eval loss. You may want to change various hyperparameters:

  * learning rate
  * choice of optimizer
  * hidden layer dimensions -- make sure your choice here makes sense given the number of training examples
  * batch size
  * num training steps
  * (anything else you can think of changing)

* Do not use the ```normalizer_fn``` arg on ```numeric_column```.

## Document
Google LLC 2018
