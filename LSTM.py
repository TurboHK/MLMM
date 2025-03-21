import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成一个简单的示例时间序列数据（假设为股票价格）
np.random.seed(42)
dates = pd.date_range('20230101', periods=100)
data = pd.DataFrame(np.random.randn(100, 1), index=dates, columns=['Price'])

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# 创建数据集：使用过去5天的数据来预测第6天
def create_dataset(data, time_step=5):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# 重塑X为LSTM的输入格式
X = X.reshape(X.shape[0], X.shape[1], 1)

# 切分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # 预测一个值

# 编译和训练模型
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 预测并逆标准化
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# 评估模型
plt.plot(data.index[train_size + 5:], data.values[train_size + 5:], label='True Data')
plt.plot(data.index[train_size + 5:], predictions, label='Predicted Data')
plt.legend()
plt.show()
