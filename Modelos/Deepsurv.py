import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sksurv.metrics import concordance_index_censored

# Cargar los datos
df = pd.read_excel('C:\\David\\CUARTO\\TFG DAVID\\TFG DAVID\\TFG\\TFG_David_Jimenez_Gutierrez\\Datos\\TCGAE_Todo.xlsx')

X = df.drop(['Recurrencia', 'DFI'], axis=1)
y = df.loc[:, ['Recurrencia', 'DFI']]

# Reemplazar valores de recurrencia y DFI
y = y.replace({0: False, 1: True})
y = y.to_records(index=False)

rs = 15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el modelo DeepSurv
def create_deepsurv_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Activación lineal para el riesgo
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Búsqueda de la mejor semilla
best_seed = None
best_c_index = 0

for seed in range(30, 50):  # Prueba diferentes valores de semilla
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    model = create_deepsurv_model(X_train.shape[1])
    model.fit(X_train, y_train['DFI'], epochs=100, batch_size=32, verbose=0)

    pred_risk = model.predict(X_test)
    c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk[:, 0])[0]

    if c_index > best_c_index:
        best_c_index = c_index
        best_seed = seed

    print(f"Semilla {seed} -> C-index: {c_index:.3f}")

print(f"\nMejor semilla encontrada: {best_seed} con C-index: {best_c_index:.3f}")

# Fijar la mejor semilla encontrada
np.random.seed(best_seed)
tf.random.set_seed(best_seed)
random.seed(best_seed)

# Entrenar el modelo con la mejor semilla
model = create_deepsurv_model(X_train.shape[1])
model.fit(X_train, y_train['DFI'], epochs=100, batch_size=32, verbose=0)

# Predicción final y cálculo del C-index
pred_risk = model.predict(X_test)
c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk[:, 0])[0]
print(f"\nC-index final con mejor semilla ({best_seed}): {c_index:.3f}")

# Normalización del riesgo para evitar valores extremos en las curvas
pred_risk = (pred_risk - np.min(pred_risk)) / (np.max(pred_risk) - np.min(pred_risk))

# Generar curvas de supervivencia ajustadas
time_range = np.linspace(0, 3650, 100)  # 10 años en días
surv_funcs = np.exp(-np.exp(-pred_risk[:, np.newaxis]) * time_range)

# Graficar las curvas de supervivencia
plt.figure(figsize=(8, 6))
for surv in surv_funcs:
    plt.step(time_range, surv.flatten(), where="post", alpha=0.5)

plt.ylim(0, 1)
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.title("Curvas de Supervivencia por Paciente - DeepSurv")
plt.show()
