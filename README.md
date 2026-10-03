# Análisis predictivo de Churn

Proyecto de la asignatura de Inteligencia Artificial (UNAB, 2024): predecir qué clientes de una empresa tienen probabilidad de abandonarla (*churn*) a partir de su comportamiento de compra y retención.

## Qué hace

1. **Carga y limpieza** del dataset `DatosEmpresaChurn.csv`: renombrado de columnas, interpolación lineal de valores nulos (`Visita`, `Categoria`) y conversión de tipos.
2. **Análisis exploratorio**: estadísticas descriptivas, histogramas y mapa de calor de correlaciones con la variable objetivo `Sefue`.
3. **Selección de variables** por correlación: `Tasa_Retencion`, `Indicador_Retencion`, `Dias_In`, `Promociones` y `Categoria`.
4. **Normalización** con `MinMaxScaler`.
5. **Entrenamiento** de un clasificador Naïve Bayes gaussiano con partición estratificada 80/20 y evaluación con *accuracy*.
6. **Persistencia** del modelo con `joblib` para reutilizarlo en predicciones nuevas.

El repositorio también incluye modelos serializados de **árbol de decisión**, **bosque aleatorio** y un cuarto clasificador (`modeloArbol.bin`, `modeloBosque.bin`, `modelobc.bin`) para comparar enfoques.

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · joblib

## Ejecutar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python "análisis_predictivo_de_churn_leandro.py"
```

El dataset se descarga automáticamente desde su URL pública.

## Autor

**Leandro Cortés** — Full Stack Developer · Ingeniería de Sistemas, UNAB
[Portafolio](https://www.leandro-cortes.com) · [LinkedIn](https://www.linkedin.com/in/leandro-cort%C3%A9s-6a0311191/)
