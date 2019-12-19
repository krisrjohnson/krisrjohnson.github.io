---
layout: single
tags: hands_on_ml, machine_learning
---

### Terms

_Chapter 1_: sampling noise, sampling bias, feature selection, feature extraction, overfitting, underfitting, regularization, degrees of freedom, hyperparameters, instance-based, model-based, training set, validation set, test set, generalization error, cost function, cross-validation

_Chapter 2_: RMSE (RMSE==Euclidean norm/distance==l2 norm), y_hat, MAE (or MAD), l_k norm, stratified sampling, standard correlation coefficient (Pearson's r), feature scaling, min-max scaling, standardization




## Chapter 2

Summary: chapter two goes through an entire ML project, from start to scratch.

The test set should be _representative of important features_. So if income is a valuable indicator of median_housing_value, then the test set needs to be representative of the different income categories. Since income is not categorical, it can be transformed into categorical with reasonable choices for buckets (look at the histogram to help decide).

An imputer can be used to calculate and populate missing values, which then needs to be used on the test set and in production.

Categorical variables need to be transformed to one hot encoded vectors, instead of serialized to an array of ints. A one hot encoded vector removes any possible similarity calculations across categories based on an integer value from serialization. Each category within the variable gets a binary column indicating either this category (1) or any of the other categories (0). Numpy's `.toarray(one_hot_encoded_matrix)` will transform the SciPy sparse matrix to a massive dense matrix if necessary.

With data loaded into a pandas dataframe, a couple of basic plots can be quickly pumped out. `df.plot(kind='scatter', x='col1', y='col2', s=df.col3/100, label='col3', figsize=(10,8), c='col4', cmap=plt.get_cmap('jet'), colorbar=True, sharex=False); plt.legend()` churns out a scatter plot with circles for datapoints whose size is determined by col3. The color of the datapoint is determined by col4 and represeneted by a heatmap, in this case 'jet'. `scatter_matrix(df[['col1','col2','col3']], figsize=(12,8))` will take continuous numerical variables and plot a matrix of scatterplots, with the diagonal being a histogram. `df.hist(bins=50)` will plot a histogram of all the numerical variables in the dataframe.

Scikit-learn has MinMaxScaler and StandardScaler transformer classes. Fit these to the training data, then _apply the same transforms to test and prod data_.

#### Pipelines and Pandas
Scikit-learn provides a `Pipeline` class, which on instantiation takes a list of name/estimator pairs, all but the last one having to be transformers. So you'd create transforms and a `Pipeline` instance for numerical columns and categorical columns, then combine those two pipelines into one preprocessing transformation step using a `ColumnsTransformer()` class:

```python
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)), #list of attrib col names
    ('imputer', SimpleImputer(strategy='median')),
    #attribs_adder is a custom transformer allowing better testing
    #the notebook has a further update github.com/ageron/handson-ml/02...ipynb
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)), #list of attrib col names
    ('one_hot_encoder', OneHotEncoder(sparse=False)),
])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline(housing) #housing is a dataFrame
```

_Scikit-Learn cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the MSE (i.e., a negative value)._


### Some Intuition on Distance
l_k norm: the l_k norm of a vector _v_ with n elements: `||v||_k = (|v_0|^k + |v_1|^k + ... + |v_n|^k)^1/k`

MAE or MADifference is also known as the Manhattan distance as it's the distance to travel in a grid city with orthogonal streets. So in a 2D situation where you can't directly traverse as the crow flies but can only choose one component direction to move in, it's the step distance.

However in a 2D situation where you can travel as the crow flies, you'd be traversing the hypotenuse, which is the L2 norm of the 2 vectors [x_0, y_0], [x_1, y_1].

The higher the norm index, the more it focuses on large values (from exponential growth).

---
Steps:
1. Looking at the big picture
2. Get the data
3. Discover and visualize data to gain insights
4. Prep data for ML algos
5. Select model and train
6. Fine-tune model
7. Present solution
8. Launch, monitor, and maintain system

---
## Chapter 1

Summary: chapter one goes over ML basics, what it is, when to use it, and the basics of training and interpreting results.


#### Q&A
1. How would you describe Machine Learning? `Machine learning is how to feed data in to a computer and finetune the computers performance on a task based on that data.`
2. Can you name four types of problems where ML shines?  `Generalized answer: supervised learning, semi-supervised learning, unsupervised learning, online ML prediction.  Specific answer: spam or not, what to watch next recommender engines, voice recognition, facial recognition`
3. What is a labeled training set?  `A set of training data which is labeled? Data a model-based ML algo would use to improve performance (as defined by a loss function, like MSE). The data is labeled, aka there is a "right answer" per datapoint.`
4. What are the two most common supervised learning tasks?  `regression and classification`
5. Can you name four unsupervised learning tasks?  `clustering, visualization, dimensionality reduction, association rule learning`
6. What ML algo would you use to allow a robot to walk in various unknown terrains?  `Reinforcement learning`
7. What ML algo would you use to segment customers?  `Clustering, or classification`
8. Is spam detection supervised or unsupervised?  `Supervised`
9. What is an online learning system?  `the algorithm continues to learn during production, such as spam detection`
10. What is out-of-core learning? `training data is too large to fit into memory, so batch learning necessary`
11. What type of learning algo relies on a similarity measure to make predictions? `instance-based`
12. What is the difference between a model parameter and a learning algo's hyperparameter? `hyperparams are tuned while model params get learned during training. The choice of hyperparam affects how well a model's params are learned.`
13. What do model-based learning algos search for? What is the most common strategy they use to succeed? How do they make predictions? `Model-based algos are used for making predictions on new data points. They're searching for the lowest error rate (cost). `
14. Can you name four of the main challenges of ML? `overfitting, underfitting, data collection, data cleaning`
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? `Overfitting. Reduce complexity, gather more data, reduce noise (fix errors/outlier handling)`
16. What is a test set and why should you use it? `Data held out from training solely to see how well the model will generalize.`
17. What is the purpose of a validation set? `To finetune hyperparameters`
18. What can go wrong if you tune hyperparams using the test set? `You'll overfit to the test set, so the model won't generalize well`
19. What is cross-validation and why would you prefer it to a validation set? `k-folds cross validation is splitting datasets into K train,val sets. Training on each and merging the end models together to avoid a sampling bias when dividing training data into train and validation.`

Wrong/incomplete answers: {5, 11}
