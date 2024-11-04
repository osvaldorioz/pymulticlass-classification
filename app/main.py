from fastapi import FastAPI
import multiclass_module
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

# Definir el modelo para el vector
class VectorI(BaseModel):
    vector: List[int]

@app.post("/multiclass")
async def mcc(epochs: int,
              learning_rate: float,
              inputs: Matrix, 
              labels: VectorI,
              sample: VectorF):
    start = time.time()

    # Datos de entrenamiento
    #inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    #labels = [0, 1, 2]  # Etiquetas para las clases
    
    #ll = len(inputs.matrix)
    size = len(labels.vector)

    #print(ll)
    # Crear un modelo con 3 características de entrada y 3 clases de salida
    model = multiclass_module.MultiClassClassifier(size, size)

    # Entrenar el modelo
    #learning_rate=0.01
    #epochs=1000
    model.train(inputs.matrix, labels.vector, learning_rate, epochs)

    # Hacer una predicción
    #input_example = [2.0, 3.0, 4.0]
    predictions = model.predict(sample.vector)
    
    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Predicciones": predictions
    }
    jj = json.dumps(j1)

    return jj