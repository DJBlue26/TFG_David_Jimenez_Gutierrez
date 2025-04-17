import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import shap

## Obtención y preparación de los datos
df = pd.read_excel('C:\David\CUARTO\TFG DAVID\TFG DAVID\TFG\TFG_David_Jimenez_Gutierrez\Datos\TCGAE_Todo.xlsx')

X = df.drop(['Recurrencia', 'DFI'], axis=1)
y = df.loc[:, ['Recurrencia', 'DFI']]

y = y.replace({0: False, 1: True})
y = y.to_records(index=False)

rs = 13
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

## Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Conversión de los datos para XGBoost
train_dmatrix = xgb.DMatrix(X_train, label=y_train['DFI'])
test_dmatrix = xgb.DMatrix(X_test, label=y_test['DFI'])

## Definición de parámetros optimizados para XGBoost Survival Model
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

## Importancia de las variables
importance = bst.get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=['Variable', 'Importancia']).sort_values(by='Importancia', ascending=False)
print("Importancia de variables:")
print(importance_df)

## Predicción y cálculo del C-index
pred_risk = bst.predict(test_dmatrix)
c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk)[0]
print(f"C-index: {c_index:.3f}")

## Cálculo de curvas de supervivencia usando Kaplan-Meier
median_risk = np.median(pred_risk)
low_risk = pred_risk <= median_risk
high_risk = pred_risk > median_risk

kmf_low = KaplanMeierFitter()
kmf_high = KaplanMeierFitter()

plt.figure(figsize=(8, 6))
kmf_low.fit(y_test['DFI'][low_risk], event_observed=y_test['Recurrencia'][low_risk], label="Bajo riesgo")
kmf_high.fit(y_test['DFI'][high_risk], event_observed=y_test['Recurrencia'][high_risk], label="Alto riesgo")

kmf_low.plot_survival_function()
kmf_high.plot_survival_function()

plt.title("Curvas de Supervivencia (Kaplan-Meier) según Riesgo - XGBoost")
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.legend()
plt.show()

## Cálculo manual de la curva de supervivencia basada en Cox
time_range = np.linspace(0, 3650, 100)  # 10 años en días
surv_funcs = np.exp(-np.exp(-pred_risk[:, np.newaxis]) * time_range)

plt.figure(figsize=(8, 6))
for surv in surv_funcs:
    plt.step(time_range, surv, where="post", alpha=0.5)

plt.ylim(0, 1)
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.title("Curvas de Supervivencia por Paciente - XGBoost (Cox Aproximado)")
plt.show()
