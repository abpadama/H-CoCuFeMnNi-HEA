#imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#load data
dataset1 = pd.read_csv('datafcc_upd_element.csv')
dataset2 = pd.read_csv('complete_element.csv')

# x and y
X = dataset1.drop(['E_Ads',], axis=1).values
y = dataset1['E_Ads'].values

#Split to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter values for tuning
constant_values = np.logspace(0.01, 3, 10) #e.g. start=10^0.01, end=10^1, 3 sample values
length_scales = np.logspace(0.01, 3, 10)
noise_levels = np.logspace(0.01, 3, 10)
alpha_values = np.logspace(-100, 0, 15)

best_mse = float('inf')
best_mae = float('inf')
best_hyperparameters = None

# Create a KFold cross-validator
cv = KFold(n_splits=5, shuffle=True, random_state=42)
i=0

# Perform hyperparameter tuning using a nested loop
for constant_value in constant_values:
    for length_scale in length_scales:
        for noise_level in noise_levels:
            for alpha in alpha_values:
                # Set current hyperparameters
                current_kernel = C(constant_value) * RBF(length_scale) + WhiteKernel(noise_level)
                model = GaussianProcessRegressor(kernel=current_kernel, alpha=alpha, random_state=42)

                # Perform cross-validation and compute mean squared error
                mse_values = []
                mae_values = []
                r2_values = []
                for train_index, val_index in cv.split(X):
                    X_train_fold, X_val = X[train_index], X[val_index]
                    y_train_fold, y_val = y[train_index], y[val_index]

                    model.fit(X_train_fold, y_train_fold)
                    y_pred_train = model.predict(X_train_fold)
                    y_pred_val = model.predict(X_val)

                    mse = mean_squared_error(y_val, y_pred_val)
                    mse_values.append(mse)

                    mae = mean_absolute_error(y_val, y_pred_val)
                    mae_values.append(mae)

                    r2_fold = r2_score(y_val,y_pred_val)
                    r2_values.append(r2_fold)

                    i += 1
                    print(i,' fit/s completed.')

                mean_mse = np.mean(mse_values)
                mean_mae = np.mean(mae_values)
                mean_r2 = np.mean(r2_values)

                # Update best hyperparameters if the current configuration is better
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_mse = mean_mse
                    best_r2 = mean_r2

                    best_hyperparameters = {
                        'constant_value': constant_value,
                        'length_scale': length_scale,
                        'noise_level': noise_level,
                        'alpha': alpha
                    }

# Train the GPR model with the best chosen hyperparameters
best_kernel = C(best_hyperparameters['constant_value']) * RBF(length_scale=best_hyperparameters['length_scale']) + WhiteKernel(noise_level=best_hyperparameters['noise_level'])
best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_hyperparameters['alpha'], random_state=42)

# # for testing only
# best_hyperparameters = {
#     'constant_value': 46.89333400125802,
#     'length_scale': 46.89333400125802,
#     'noise_level': 21.821715475664465,
#     'alpha': 5.179474679231308e-15
# }
# best_kernel = C(best_hyperparameters['constant_value']) * RBF(length_scale=best_hyperparameters['length_scale']) + WhiteKernel(noise_level=best_hyperparameters['noise_level'])
# best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_hyperparameters['alpha'], random_state=42)

best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# save E_ads predicted for the whole dataset
y_pred_dataset = best_model.predict(dataset1.drop(['E_Ads',], axis=1).values)
dataset_pred = dataset1
dataset_pred['E_ads_pred'] = y_pred_dataset
dataset_pred.to_csv('dataset-fcc-element.csv')

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train,y_train_pred)
r2_test = r2_score(y_test,y_test_pred)

# Print the best hyperparameters and corresponding mean squared error
print("Best Hyperparameters:", best_hyperparameters)
print("Train Mean Absolute Error:", mae_train)
print("Train Root Mean Squared Error:", np.sqrt(mse_train))
print("Train R2:", r2_train)
print("Test Mean Absolute Error:", mae_test)
print("Test Root Mean Squared Error:", np.sqrt(mse_test))
print("Test R2:", r2_test)

#plot
plt.rcParams.update({'font.size': 12})
plt.scatter(y_train,y_train_pred,s=15, color='blue', label='Train')
plt.scatter(y_test,y_test_pred,s=15, color='red', label='Test')

# Calculate the slope of the line (change in y / change in x)
slope = (y_test[-1] - y_test[0]) / (y_test[-1] - y_test[0])
# Extend the lines by creating new x-values
x_extended = np.linspace(-1, 1, 100)  # Generate 100 new x-values from -1 to 1
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
plt.xlim(-0.6, 0.1)
plt.ylim(-0.6, 0.1)
plt.legend()
plt.rcParams['figure.dpi']=300
plt.savefig('gpr-fcc-element.jpg')

#predict remaining/other sites
X_all = dataset2.values

y_pred_all = best_model.predict(X_all)
print(y_pred_all)

dataset2['E_ads'] = y_pred_all
print(dataset2)

dataset2.to_csv('complete-fcc-element.csv')
