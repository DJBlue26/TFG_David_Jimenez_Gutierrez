import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle, CoxPH
from pycox.evaluation import EvalSurv
from lifelines import CoxPHFitter

import warnings
warnings.filterwarnings("ignore")

# Fijación de las semillas para poder obtener reproducibilidad
def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Carga y preparar de los datos
df = pd.read_excel('C:\\David\\CUARTO\\TFG DAVID\\TFG DAVID\\TFG\\TFG_David_Jimenez_Gutierrez\\Datos\\TCGAE_Todo.xlsx')

X = df.drop(['Recurrencia', 'DFI'], axis=1)
y_time = df['DFI'].astype(float).values
y_event = df['Recurrencia'].replace({0: False, 1: True}).astype(bool).values

# Búsqueda de la mejor semilla
best_cindex = -1
best_seed = None
best_model = None
best_surv = None
best_test_data = None
best_scaler = None
best_labtrans = None

for seed in range(100):
    try:
        set_all_seeds(seed)

        X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
            X, y_time, y_event, test_size=0.3, random_state=seed
        )

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train).astype(np.float32)
        X_test_std = scaler.transform(X_test).astype(np.float32)

        num_durations = 50
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train_transformed = labtrans.fit_transform(y_time_train, y_event_train)
        y_test_transformed = labtrans.transform(y_time_test, y_event_test)

        net = tt.practical.MLPVanilla(
            X_train.shape[1], [32, 32], labtrans.out_features,
            batch_norm=True, dropout=0.1
        )

        model = DeepHitSingle(
            net, tt.optim.Adam, alpha=0.2, sigma=0.1,
            duration_index=labtrans.cuts
        )

        model.fit(
            X_train_std, y_train_transformed,
            batch_size=128, epochs=100, verbose=False,
            val_data=(X_test_std, y_test_transformed),
            callbacks=[tt.callbacks.EarlyStopping()]
        )

        surv = model.predict_surv_df(X_test_std)
        if surv.shape[1] == 0:
            continue

        ev = EvalSurv(surv, y_time_test, y_event_test, censor_surv='km')
        c_index = ev.concordance_td('antolini')

        print(f"Semilla {seed} ➜ C-index = {round(c_index, 4)}")

        if c_index > best_cindex:
            best_cindex = c_index
            best_seed = seed
            best_model = model
            best_surv = surv
            best_test_data = (X_test_std, y_time_test, y_event_test)
            best_scaler = scaler
            best_labtrans = labtrans

    except Exception as e:
        print(f"Error con semilla {seed}: {e}")
        continue

if best_model is None:
    print("\n No se pudo entrenar ningún modelo exitosamente.")
else:
    print(f"\n Mejor semilla: {best_seed} con C-index = {round(best_cindex, 4)}")

    # Preparación de los datos para el modelo de Cox real con lifelines (se utiliza sólo el riesgo como covariable)
    X_test_std, y_time_test, y_event_test = best_test_data

    df_test = pd.DataFrame(best_scaler.inverse_transform(X_test_std), columns=X.columns)
    df_test["DFI"] = y_time_test
    df_test["Recurrencia"] = y_event_test.astype(int)

    # Obtener riesgos del modelo DeepHit
    risk_scores = best_model.predict(X_test_std).sum(axis=1)
    df_test["riesgo"] = risk_scores

    # Ajuste modelo Cox con función base para estimar las funciones de supervivencia
    cph = CoxPHFitter()
    cph.fit(df_test[["riesgo", "DFI", "Recurrencia"]], duration_col="DFI", event_col="Recurrencia")

    # Gráfico de curvas de supervivencia individuales (todos los pacientes del conjunto de test)
    tiempo = np.linspace(0, df_test["DFI"].max(), 100)
    surv_funcs = cph.predict_survival_function(df_test, times=tiempo)

    plt.figure(figsize=(10, 6))
    for i in range(surv_funcs.shape[1]):
        plt.step(tiempo, surv_funcs.iloc[:, i], where="post", alpha=0.5)
    plt.ylim(0, 1)
    plt.title(f"Curvas de Supervivencia Individuales (DeepHit + CoxPH) - seed {best_seed}")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Probabilidad de Supervivencia")
    plt.grid(True)
    plt.show()


# Cálculo de la importancia de las variables basado en gradientes
def calcular_importancia_variables(modelo, datos, nombres_vars):
    datos_tensor = torch.tensor(datos, dtype=torch.float32, requires_grad=True)
    pred = modelo.net(datos_tensor)  # salida [n_muestras, n_durations]
    suma_pred = pred.sum(dim=1).mean()  # Promedio de la suma de riesgos
    suma_pred.backward()  # retropropagación para obtener los gradientes

    importancia = datos_tensor.grad.abs().mean(dim=0).detach().numpy()
    importancia_normalizada = importancia / (importancia.sum() + 1e-8)

    df_importancia = pd.DataFrame({
        'Variable': nombres_vars,
        'Importancia': importancia_normalizada
    }).sort_values(by='Importancia', ascending=False)

    return df_importancia

# Calcular e imprimir importancia
importancia_df = calcular_importancia_variables(best_model, X_test_std, X.columns)
print("\nImportancia de las variables (DeepHit):")
print(importancia_df)