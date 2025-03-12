import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

## Obtención y preparación de los datos
df = pd.read_excel('C:\David\CUARTO\TFG DAVID\TFG DAVID\TFG\TFG_David_Jimenez_Gutierrez\Datos\TCGAE_Todo.xlsx')

X = df.drop(['Recurrencia','DFI'], axis=1)
y = df.loc[:, ['Recurrencia', 'DFI']]

y = y.replace({0: False, 1: True})
y = y.to_records(index=False)

rs = 15
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

## Predicción y cálculo del C-index
pred_risk = bst.predict(test_dmatrix)
c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk)[0]
print(f"C-index: {c_index:.3f}")

## Curvas de supervivencia
surv_times = np.sort(y_test['DFI'])
surv_preds = np.exp(-np.exp(-pred_risk))  # Transformación correcta para supervivencia

plt.step(surv_times, surv_preds, where="post", label="XGBoost Survival Curve")
plt.ylim(0, 1)
plt.xlabel("Tiempo")
plt.ylabel("Probabilidad de Supervivencia")
plt.title("Curva de Supervivencia - XGBoost")
plt.legend()
plt.show()
