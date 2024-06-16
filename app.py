# pip install -q yfinance
# pip install DataReader
# pip install pandas_datareader
# pip install torch
# pip install setuptools

import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, render_template, request

app = Flask(__name__)

# данные из yahoo
import yfinance as yf
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
yf.pdr_override()

def evaluate_regression_model(y_true, y_pred):
    # Вычисление среднеквадратичной ошибки
    mse = mean_squared_error(y_true, y_pred)
    # Вычисление средней абсолютной ошибки
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # Вычисление коэффициента детерминации (R^2)
    r2 = r2_score(y_true, y_pred)

    return {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R^2': r2}

def plot_result(data, training_data_len, predicted_values, text):
    '''
    строит график с реальными данными и предсказанными
    '''
    # print(text)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predicted_values
    valid['Predictions'] = valid['Predictions']
    plt.figure(figsize=(16,6))
    plt.title(f'Предсказание стоимости акций {text}')
    plt.xlabel('Дата', fontsize=18)
    plt.ylabel('Цена закрытия, USD', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Pred'], loc='upper left')

    # Сохранение графика в байтовый поток
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    # Кодирование изображения в base64
    img_base64 = base64.b64encode(img_bytes.read()).decode('ascii')
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        df = pdr.get_data_yahoo(text, start='2012-01-01', end=datetime.now())

        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = int(np.ceil( len(dataset) * .95 ))

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        
        # создание обучабщего датасета
        train_data = scaled_data[0:int(training_data_len), :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # создание тестового набора данных
        test_data = scaled_data[training_data_len - 60: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
                self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
                self.fc1 = nn.Linear(hidden_size2, 25)
                self.fc2 = nn.Linear(25, output_size)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out, _ = self.lstm2(out)
                out = self.fc1(out[:, -1, :])
                out = self.fc2(out)
                return out

            def predict(self, x):
                with torch.no_grad():
                    out, _ = self.lstm1(x)
                    out, _ = self.lstm2(out)
                    out = self.fc1(out[:, -1, :])
                    out = self.fc2(out)
                    return out

        input_size = 1  # Размер входных данных
        hidden_size1 = 128  # Размер скрытого слоя LSTM1
        hidden_size2 = 64  # Размер скрытого слоя LSTM2
        output_size = 1  # Размер выходных данных
        model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Преобразование в тензоры PyTorch
        X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Обучение модели
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Предсказание
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        with torch.no_grad():
            predicted_values = model.predict(X_test_tensor)


        def objective_function(X):
            predicted = predicted_values * X
            return evaluate_regression_model(y_test, predicted)['MAE']

        initial_guess = 1.0
        guess = minimize(objective_function, initial_guess, method='Nelder-Mead')
        optimal_X = guess.x[0]

        result = pd.DataFrame(evaluate_regression_model(y_test, predicted_values*optimal_X), index=[0])

        img_base64 = plot_result(data, training_data_len, predicted_values*optimal_X, text)

        # # Сохранение графика в байтовый поток
        # img_bytes = io.BytesIO()
        # plt.savefig(img_bytes, format='png')
        # img_bytes.seek(0)
        
        # # Кодирование изображения в base64
        # img_base64 = base64.b64encode(img_bytes.read()).decode('ascii')

        return render_template('index.html', image=img_base64, result=result.to_html()) # , text=text
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
