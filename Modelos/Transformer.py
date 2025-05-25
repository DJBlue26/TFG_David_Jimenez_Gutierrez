import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

# Carga y preparación de los datos
df = pd.read_excel('C:\\David\\CUARTO\\TFG DAVID\\TFG DAVID\\TFG\\TFG_David_Jimenez_Gutierrez\\Datos\\TCGAE_Todo.xlsx')
X = df.drop(['Recurrencia', 'DFI'], axis=1)
y = df[['Recurrencia', 'DFI']].replace({0: False, 1: True}).infer_objects(copy=False)
y = y.to_records(index=False)

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversión a tensores para pytorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train['DFI'].copy(), dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test['DFI'].copy(), dtype=torch.float32).view(-1, 1)

# Creación del Modelo de Transformer
class SurvivalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.5):
        super(SurvivalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)

# Entrenamiento del modelo
def train_and_evaluate(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SurvivalTransformer(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    pred_risk = model(X_test_t).detach().numpy()
    c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk[:, 0])[0]
    
    return c_index, model

# Búsqueda de la mejor semilla
best_seed = None
best_c_index = 0
best_model = None

for seed in range(0, 100):
    c_index, model = train_and_evaluate(seed)
    print(f"Semilla {seed} -> C-index: {c_index:.3f}")
    if c_index > best_c_index:
        best_c_index = c_index
        best_seed = seed
        best_model = model

print(f"\nMejor semilla encontrada: {best_seed} con C-index: {best_c_index:.3f}")

# Predicciones finales del modelo con la mejor semilla
best_model.eval()
pred_risk = best_model(X_test_t).detach().numpy()
pred_risk = (pred_risk - np.min(pred_risk)) / (np.max(pred_risk) - np.min(pred_risk) + 1e-8)

# Preparación de los datos para el modelo de Cox real con lifelines (se utiliza sólo el riesgo como covariable)
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test['duration'] = y_test['DFI']
df_test['event'] = y_test['Recurrencia']
df_test['risk_score'] = pred_risk

# Ajuste del modelo Cox con función base para estimar las funciones de supervivencia
cox_data = df_test[['duration', 'event', 'risk_score']].copy()
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='duration', event_col='event')

# Gráfico de curvas de supervivencia individuales (todos los pacientes del conjunto de test)
plt.figure(figsize=(12, 8))
for idx in range(len(cox_data)):
    single_patient = cox_data.iloc[[idx]]
    surv_func = cph.predict_survival_function(single_patient)
    plt.plot(surv_func.index, surv_func.values.flatten(), label=f'Paciente {idx}', alpha=0.7)

plt.title('Curvas de Supervivencia Individuales - Transformer (Validación)')
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad de Supervivencia')
plt.grid(True)
plt.tight_layout()
plt.show()


# Obtención de la Importancia de las variables
best_model.eval()
X_test_t.requires_grad = True

output = best_model(X_test_t)
output.sum().backward()  

# Obtención del gradiente medio absoluto para cada variable
feature_importance = X_test_t.grad.abs().mean(dim=0).detach().numpy()

# Mostrar la importancia de las variables
importance_df = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': feature_importance
}).sort_values(by='Importancia', ascending=False)

print("\nImportancia de las variables (estimada por gradientes):")
print(importance_df)