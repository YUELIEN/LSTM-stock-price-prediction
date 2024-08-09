import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time

df = yf.Ticker('2330.TW').history(period='max')
df = df.filter(['Close'])
df = df.rename(columns={'Close':'GT'})
df['GT'] = df['GT'].rolling(window=5).mean()
df = df.dropna()

plt.style.use('seaborn-v0_8-darkgrid')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df['GT'], linewidth=1)
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
scaler_prices = scaler.fit_transform(df.values)
print(scaler_prices)

MOVING_WIN_SIZE = 60

all_x, all_y = [], []
for i in range(len(scaler_prices) - MOVING_WIN_SIZE):
    x = scaler_prices[i:i+MOVING_WIN_SIZE]
    y = scaler_prices[i+MOVING_WIN_SIZE]
    all_x.append(x)
    all_y.append(y)

all_x, all_y = np.array(all_x), np.array(all_y)
print(all_x.shape)
print(all_y.shape)

DS_SPLIT = 0.8

train_ds_size = round(all_x.shape[0] * DS_SPLIT)
train_x, train_y = all_x[:train_ds_size], all_y[:train_ds_size]
test_x, test_y = all_x[train_ds_size:], all_y[train_ds_size:]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.summary()

learning_rate = 5e-5
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

# callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', verbose=2, patience=200)
start_time = time.time()
history = model.fit(train_x, train_y,
          validation_split=0.2,
          # callbacks=[callback],
          epochs=1000,
          batch_size=8)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Loss and MAE')
plt.xlabel('Epoch')
plt.ylabel('Loss / MAE')
plt.legend(loc='upper right')
plt.show()

preds = model.predict(test_x)
preds = scaler.inverse_transform(preds)

train_df = df[:train_ds_size+MOVING_WIN_SIZE]
test_df = df[train_ds_size+MOVING_WIN_SIZE:]
test_df = test_df.assign(Predict=preds)

plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train_df['GT'], linewidth=2)
plt.plot(test_df['GT'], linewidth=2)
plt.plot(test_df['Predict'], linewidth=1)
plt.legend(['Train', 'GT', 'Predict'])
plt.show()

start_date = '2024-01-1'
subset = test_df.loc[start_date:]
plt.plot(subset.index, subset['GT'], linewidth=2)
plt.plot(subset.index, subset['Predict'], linewidth=1)
plt.legend(['GT', 'Predict'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

test_df = test_df.assign(Shifted=test_df['GT'].shift(1))
test_df.iat[0,-1] = train_df.iat[-1,-1]

predict_rmse = mean_squared_error(test_df['GT'], test_df['Predict'], squared=False)
predict_mae = mean_absolute_error(test_df['GT'], test_df['Predict'])
predict_mape = mean_absolute_percentage_error(test_df['GT'], test_df['Predict'])
predict_cvrmse = predict_rmse / test_df['GT'].mean() * 100
predict_r2 = r2_score(test_df['GT'], test_df['Predict'])

print(f"Predict RMSE: {predict_rmse}")
print(f"Predict MAE: {predict_mae}")
print(f"Predict MAPE: {predict_mape}")
print(f"Predict CVRMSE: {predict_cvrmse}%")
print(f"Predict R2: {predict_r2}")

shifted_rmse = mean_squared_error(test_df['GT'], test_df['Shifted'], squared=False)
shifted_cvrmse = shifted_rmse / test_df['GT'].mean() * 100

print(f"Shifted RMSE: {shifted_rmse}")
print(f"Shifted CVRMSE: {shifted_cvrmse}%")
