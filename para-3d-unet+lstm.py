import os
import tensorflow as tf
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#-----------------------------------------------
# S. Maishal
# email :: subhadeepmaishal@kgpian.iitkgp.ac.in
#-----------------------------------------------
# Set environment variables for parallelism (400 cores)
os.environ['OMP_NUM_THREADS'] = '400'
os.environ['TF_NUM_INTRAOP_THREADS'] = '400'
os.environ['TF_NUM_INTEROP_THREADS'] = '400'

# Limit TensorFlow to use 40 cores for parallel computation
tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

# Function to replace NaN values with the mean of each variable
def replace_nan_with_mean(data):
    nan_mean = np.nanmean(data)
    data[np.isnan(data)] = nan_mean
    return data

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/Unet/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
chl = ds['chl'].values  # Target variable 
depth = ds['depth'].values  # if you find any  depths here.

# Replace NaN values in predictors and target using replace_nan_with_mean
fe = replace_nan_with_mean(fe)
po4 = replace_nan_with_mean(po4)
si = replace_nan_with_mean(si)
no3 = replace_nan_with_mean(no3)
chl = replace_nan_with_mean(chl)

# Stack the input variables along a new channel dimension (fe, po4, si, no3)
inputs = np.stack([fe, po4, si, no3], axis=-1)

# Prepare input for LSTM
time_steps = 5
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])
y_lstm = chl[time_steps:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

# Define the U-Net + LSTM Model 3D ( i think you lovely, can fix it? )
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))(inputs)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D((2, 2, 2)))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D((2, 2, 2)))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling3D())(x)
x = tf.keras.layers.LSTM(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(y_train.shape[1] * y_train.shape[2] * y_train.shape[3])(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled.reshape(y_train_scaled.shape[0], -1), 
                    epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test_scaled.reshape(y_test_scaled.shape[0], -1))

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1] * y_test.shape[2] * y_test.shape[3])).reshape(y_test.shape)

# Replace NaN values in the predicted output
predicted_y = replace_nan_with_mean(predicted_y)

# Define output file path
output_file_path = r"/scratch/20cl91p02/ANN_BIO/Unet/average_output_unet+lstm_chl.nc"

# Save to NetCDF
ds_out = xr.Dataset(
    {
        'predicted_chl': (('time', 'latitude', 'longitude', 'depth'), predicted_y)
    },
    coords={'time': ds['time'].values, 'latitude': ds['latitude'].values, 'longitude': ds['longitude'].values, 'depth': depth}
)
ds_out.to_netcdf(output_file_path)

print("Output saved to NetCDF:", output_file_path)
