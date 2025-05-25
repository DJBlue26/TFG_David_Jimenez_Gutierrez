Este repositorio se basa en el desarrollo de 4 modelos predictivos en Python para analizar la recurrencia en pacientes de cáncer de mama. 
Se va a evular la precisión de los modelos mediante una métrica llamada c-index o índice de concordancia y por otra parte mediante la
representación de las curvas de supervivencia individuales de cada paciente basadas en las prediciciones de cada modelo.

La carpeta "Datos" contiene los conjuntos de datos que han sido utilizados:

    - TCGAE_Clinico: Contiene información únicamente acerca de las características clínicas de los pacientes.
    
    - TCGAE_Rad: Contiene información únicamente acerca de las características radiómicas obtenidas de las MRI de los pacientes.
    
    - TCGAE_Todo: Contiene el conjunto de datos final, que ha sido utilizado para el entrenamiento de los modelos predictivos, 
                  que combina los datos seleccionados de los conjuntos anteriores manteniendo en todo momento la privacidad
                  de los pacientes ya que no aparecen sus nombres y se utilizan etiquetas simplificadas.

La carpeta "Modelos" contiene los 4 modelos predictivos implementados que han utilizado el conjutno de datos anterior:

    - XGBoost.py: Se trata de la técnica de Extreme Gradient Boosting desarrollada para analizar la recurrencia.

    - DeepSurv.py: Se trata de la red neuronal profunda basada en Cox desarrollada para analizar la recurrencia.

    - Transformer: Se trata de la arquitectura Transformer, en concreto Transformer encoder, adaptada para realizar
                   un análisis de supervivencia.
                   
    - DeepHit.py: Se trata de la red neuronal profunda basada en riegos desarrollada para analizar la recurrencia.
