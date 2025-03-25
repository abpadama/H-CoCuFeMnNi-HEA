#imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#load data
dataset1 = pd.read_csv('data_h_fcc_element.csv') # change file name
dataset2 = pd.read_csv('combinations_h_fcc.csv') # change file name

# x and y
X = dataset1.drop(['E_ads',], axis=1).values
y = dataset1['E_ads'].values

#Split to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80, random_state=42)

gpr = GaussianProcessRegressor(random_state=0)


# Define hyperparameter values for tuning
constant_values = np.linspace(10, 300, 15)  # Bound values for constant kernel
length_scales = np.linspace(40, 300, 15)    # Bound values for RBF kernel length scale
noise_levels = np.linspace(1, 300, 15)      # Bound values for White kernel noise level
alpha_values = np.logspace(-10, 0, 15)  # Smaller range of alpha for regularization

# Parameter grid for hyperparameter tuning
gpr_param_grid = [
    {
        'alpha': alpha_values,
        'kernel': [
            ConstantKernel(constant_value=c_val) *
            RBF(length_scale=l_val) +
            WhiteKernel(noise_level=n_val) for c_val in constant_values for l_val in length_scales for n_val in noise_levels
        ]
    }
]

# Grid Search with neg_mean_absolute_error as the scoring metric
gpr_search = GridSearchCV(gpr, gpr_param_grid, n_jobs=-1, cv=5,
                          scoring='neg_mean_absolute_error', verbose=0)

# Fit the GridSearchCV with training data
gpr_search.fit(X_train, y_train)

best_gpr = gpr_search.best_params_

# Train the GPR model with the best chosen hyperparameters
y_train_pred = gpr_search.best_estimator_.predict(X_train)
y_test_pred = gpr_search.best_estimator_.predict(X_test)

# save E_ads predicted for the whole dataset
y_pred_dataset = gpr_search.best_estimator_.predict(dataset1.drop(['E_ads',], axis=1).values)
dataset_pred = dataset1
dataset_pred['E_ads_pred'] = y_pred_dataset

dataset_pred.to_csv('dataset-fcc-element.csv') # change file name

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train,y_train_pred)
r2_test = r2_score(y_test,y_test_pred)

# Print the best hyperparameters and corresponding mean squared error
print("Best Hyperparameters:", best_gpr)
print("Train Mean Absolute Error:", mae_train)
#print("Train Root Mean Squared Error:", np.sqrt(mse_train))
print("Train R2:", r2_train)
print("Test Mean Absolute Error:", mae_test)
#print("Test Root Mean Squared Error:", np.sqrt(mse_test))
print("Test R2:", r2_test)

#plot
plt.rcParams.update({'font.size': 12})
plt.scatter(y_train,y_train_pred,s=15, color='blue', label='Train')
plt.scatter(y_test,y_test_pred,s=15, color='red', label='Test')

# Calculate the slope of the line (change in y / change in x)
slope = (y_test[-1] - y_test[0]) / (y_test[-1] - y_test[0])
# Extend the lines by creating new x-values
x_extended = np.linspace(-2.1, -0.1, 100)  # Generate 100 new x-values from -1 to 1
# Calculate corresponding y-values for the extended line using the slope
y_extended = slope * (x_extended - y_test[0]) + y_test[0]

# offset
y_offset_pos = y_extended + 0.1  # Positive offset
y_offset_neg = y_extended - 0.1  # Negative offset

plt.plot(x_extended, y_extended, color='cyan',  linestyle='-')
plt.plot(x_extended, y_offset_pos, color='cyan',  linestyle='--')
plt.plot(x_extended, y_offset_neg, color='cyan',  linestyle='--')
plt.xlabel('${\Delta}$E$_{DFT}$' + '  ' + '(eV)')
plt.ylabel('${\Delta}$E$_{pred}$' + '  ' + '(eV)')
plt.xlim(-2.1, -0.1) # change range
plt.ylim(-2.1, -0.1) # change range
plt.legend()
plt.rcParams['figure.dpi']=300
plt.savefig('gpr-fcc-element.jpg')

#predict remaining/other sites
X_all = dataset2.values

y_pred_all = gpr_search.best_estimator_.predict(X_all)
print(y_pred_all)

dataset2['E_ads'] = y_pred_all
print(dataset2)

dataset2.to_csv('complete-fcc-element.csv') # change file name
