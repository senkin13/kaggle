from sklearn.model_selection import GridSearchCV
estimator = lgb.LGBMRegressor()
param_grid = {
    'num_leaves': [31, 33, 63, 65, 127, 129],
    'learning_rate': [0.02, 0.1, 0.03],
    'min_data_in_leaf': [250,300]
}
y_t=y_train[:, i]
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_t)

print('Best parameters found by grid search are:', gbm.best_params_)
