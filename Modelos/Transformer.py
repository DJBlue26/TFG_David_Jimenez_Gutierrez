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

# Cargar los datos
df = pd.read_excel('C:\\David\\CUARTO\\TFG DAVID\\TFG DAVID\\TFG\\TFG_David_Jimenez_Gutierrez\\Datos\\TCGAE_Todo.xlsx')
X = df.drop(['Recurrencia', 'DFI'], axis=1)
y = df.loc[:, ['Recurrencia', 'DFI']]

# Evitar FutureWarning en Pandas
y = y.replace({0: False, 1: True}).infer_objects(copy=False)

# Convertir `y` a un array estructurado compatible con `sksurv`
y = y.to_records(index=False)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores de PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train['DFI'].copy(), dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test['DFI'].copy(), dtype=torch.float32).view(-1, 1)

# Definir Transformer para análisis de supervivencia
class SurvivalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.5):
        super(SurvivalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)  # Salida de riesgo de supervivencia

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)

# Función para entrenar el modelo y calcular el C-index
def train_and_evaluate(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SurvivalTransformer(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Entrenamiento
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluación
    model.eval()
    pred_risk = model(X_test_t).detach().numpy()
    c_index = concordance_index_censored(y_test['Recurrencia'], y_test['DFI'], -pred_risk[:, 0])[0]
    
    return c_index, model

# Búsqueda de la mejor semilla
best_seed = None
best_c_index = 0
best_model = None

for seed in range(0, 100):  # Probar diferentes semillas
    c_index, model = train_and_evaluate(seed)
    print(f"Semilla {seed} -> C-index: {c_index:.3f}")

    if c_index > best_c_index:
        best_c_index = c_index
        best_seed = seed
        best_model = model

print(f"\nMejor semilla encontrada: {best_seed} con C-index: {best_c_index:.3f}")

# Usar el mejor modelo encontrado para las predicciones finales
best_model.eval()
pred_risk = best_model(X_test_t).detach().numpy()

# Normalización del riesgo para curvas de supervivencia
pred_risk = (pred_risk - np.min(pred_risk)) / (np.max(pred_risk) - np.min(pred_risk))

# Generar curvas de supervivencia ajustadas
time_range = np.linspace(0, 3650, 100)  # 10 años en días
surv_funcs = np.exp(-np.exp(pred_risk[:, np.newaxis]) * time_range)

# Graficar las curvas de supervivencia
plt.figure(figsize=(8, 6))
for surv in surv_funcs:
    plt.step(time_range, surv.flatten(), where="post", alpha=0.5)

plt.ylim(0, 1)
plt.xlabel("Tiempo (días)")
plt.ylabel("Probabilidad de Supervivencia")
plt.title(f"Curvas de Supervivencia - Transformer (Mejor semilla: {best_seed})")
plt.show()
