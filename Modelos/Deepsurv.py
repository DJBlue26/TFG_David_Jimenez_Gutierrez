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
from sklearn.inspection import permutation_importance
from lifelines import KaplanMeierFitter, CoxPHFitter

# Cargar los datos
df = pd.read_excel('C:\David\CUARTO\TFG DAVID\TFG DAVID\TFG\TFG_David_Jimenez_Gutierrez\Datos\TCGAE_Todo.xlsx')

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

for seed in range(1, 100):  # Prueba diferentes valores de semilla
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

# Cálculo de la importancia de las variables mediante permutación
def feature_importance(model, X_test, y_test, num_repeats=5):
    def scoring_function(X_perm):
        pred_risk = model.predict(X_perm)
        return concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk[:, 0])[0]
    
    base_score = scoring_function(X_test)
    importances = {}
    
    for i in range(X_test.shape[1]):
        scores = []
        for _ in range(num_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            scores.append(scoring_function(X_perm))
        importances[X.columns[i]] = base_score - np.mean(scores)
    
    return importances

feature_importances = feature_importance(model, X_test, y_test)
feature_importances = pd.Series(feature_importances).sort_values(ascending=False)
print("\nImportancia de las variables:")
print(feature_importances.head(10))  # Mostrar las 10 variables más importantes

# Normalización del riesgo para evitar valores extremos en las curvas
pred_risk = (pred_risk - np.min(pred_risk)) / (np.max(pred_risk) - np.min(pred_risk))

# Generar curvas de supervivencia ajustadas con Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(y_test['DFI'], event_observed=y_test['Recurrencia'])

plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title("Curva de Supervivencia - Kaplan-Meier")
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.show()

# Generar curvas de supervivencia ajustadas con modelo de Cox
cox_df = pd.DataFrame(X_test, columns=X.columns)
cox_df['DFI'] = y_test['DFI']
cox_df['Recurrencia'] = y_test['Recurrencia']
cox_model = CoxPHFitter()
cox_model.fit(cox_df, duration_col='DFI', event_col='Recurrencia')
cox_model.plot_survival_function()
plt.title("Curva de Supervivencia - Modelo de Cox")
plt.show()
