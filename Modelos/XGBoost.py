import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

# Carga y preparación de los datos
df = pd.read_excel('C:\\David\\CUARTO\\TFG DAVID\\TFG DAVID\\TFG\\TFG_David_Jimenez_Gutierrez\\Datos\\TCGAE_Todo.xlsx')

X = df.drop(['Recurrencia', 'DFI'], axis=1)
y = df.loc[:, ['Recurrencia', 'DFI']]

y = y.replace({0: False, 1: True})
y = y.to_records(index=False)

rs = 13

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversión de los datos para XGBoost
train_dmatrix = xgb.DMatrix(X_train, label=y_train['DFI'])
test_dmatrix = xgb.DMatrix(X_test, label=y_test['DFI'])

# Creación del modelo de XGBoost y entrenamiento del modelo
params = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "learning_rate": 0.1,
    "max_depth": 3,
    "lambda": 1.0,
    "alpha": 0.5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": rs,
}

num_round = 300
bst = xgb.train(params, train_dmatrix, num_round)

# Obtención de la Importancia de las variables
importance = bst.get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=['Variable', 'Importancia']).sort_values(by='Importancia', ascending=False)

# Establecer los nombres reales de las variables
feature_map = {f"f{i}": col for i, col in enumerate(X.columns)}
importance_df['Variable_real'] = importance_df['Variable'].map(feature_map)

# Mostrar la importancia con nombres reales
print("Importancia de variables:")
print(importance_df[['Variable', 'Variable_real', 'Importancia']])

# Predicción de riesgos
pred_risk = bst.predict(test_dmatrix)

# Cálculo del C-index
c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk)[0]
print(f"C-index: {c_index:.3f}")

# Preparación de los datos para el modelo de Cox real con lifelines (se utiliza sólo el riesgo como covariable)
df_cox = pd.DataFrame({
    "risk": pred_risk,
    "DFI": y_test['DFI'],
    "Recurrencia": y_test['Recurrencia']
})

# Ajuste modelo Cox con función base para estimar las funciones de supervivencia
cph = CoxPHFitter()
cph.fit(df_cox, duration_col="DFI", event_col="Recurrencia")

# Gráfico de curvas de supervivencia individuales (todos los pacientes del conjunto de test)
plt.figure(figsize=(10, 6))
for _, row in df_cox.iterrows():
    surv_func = cph.predict_survival_function(row.to_frame().T)
    plt.plot(surv_func.index, surv_func.values.flatten(), alpha=0.3)

plt.title("Curvas de Supervivencia Individuales (Estimadas con Cox real)")
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
