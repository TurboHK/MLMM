import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 生成一个示例时间序列数据
np.random.seed(42)
dates = pd.date_range('20230101', periods=100)
data = pd.DataFrame(np.random.randn(100, 1), index=dates, columns=['Price'])

# 创建特征和标签
def create_features(data, lag=1):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data.iloc[i:i + lag].values.flatten())  # 创建特征（过去几天的价格）
        y.append(data.iloc[i + lag].values[0])  # 创建标签（未来一天的价格）
    return np.array(X), np.array(y)

lag = 5  # 使用过去5天的数据来预测未来的价格
X, y = create_features(data, lag)

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 初始化XGBoost回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 预测未来数据
# 假设我们要预测未来5天的价格
last_window = data['Price'].iloc[-lag:].values.reshape(1, -1)
last_window = scaler.transform(last_window)

future_predictions = []
for _ in range(5):  # 预测5天
    prediction = model.predict(last_window)
    future_predictions.append(prediction[0])
    last_window = np.roll(last_window, -1)
    last_window[0, -1] = prediction[0]

print("Future Predictions:", future_predictions)
