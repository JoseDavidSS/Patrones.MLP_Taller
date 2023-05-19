import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

class PerceptronMulticapa:
    def __init__(self, capas, funciones_activacion, alpha=0.1, epochs=1000):
        self.capas = capas
        self.funciones_activacion = funciones_activacion
        self.funcion_actual = "s"
        self.alpha = alpha
        self.epochs = epochs
        self.tiempo_entrenamiento = 0.0
        self.num_capas = len(capas)
        self.bias = []
        self.pesos = []
        self.activaciones = []
        self.deltas = []

        for i in range(1, self.num_capas):
            peso = np.random.randn(self.capas[i-1], self.capas[i])
            self.pesos.append(peso)
            bias = np.random.randn(self.capas[i])
            self.bias.append(bias)

    def activacion(self, x):
        # Función de activación sigmoide
        return 1.0 / (1 + np.exp(-x))

    def activacion_derivada(self, x):
        # Derivada de la función de activación sigmoide
        return x * (1 - x)

    def activacion_tanh(self, x):
        # Función de activación tanh (tangente hiperbólica)
        return np.tanh(x)

    def activacion_derivada_tanh(self, x):
        # Derivada de la función de activación tanh (tangente hiperbólica)
        return 1 - np.tanh(x)**2
    
    def activacion_relu(self, x):
        # Función de activación ReLU (Rectified Linear Unit)
        return np.maximum(0, x)

    def activacion_derivada_relu(self, x):
        # Derivada de la función de activación ReLU (Rectified Linear Unit)
        return np.where(x <= 0, 0, 1)

    def feedforward(self, X):
        self.activaciones = [X]
        for i in range(self.num_capas - 1):
            print ("Voy a calcular la salida de la capa actual*******")
            print ("Capa actual_____________________________________:", self.capas[i])
            print ("Estoy usando la función_________________________:", self.funciones_activacion[i])
            

            # Calcular la salida de la capa actual
            if self.funciones_activacion[i] == "r": # Para ReLU
                activacion = self.activacion_relu(np.dot(self.activaciones[i], self.pesos[i]) + self.bias[i])

            elif self.funciones_activacion[i] == "t": # Para tanh
                activacion = self.activacion_tanh(np.dot(self.activaciones[i], self.pesos[i]) + self.bias[i])

            else: # Para sigmoide
                activacion = self.activacion(np.dot(self.activaciones[i], self.pesos[i]) + self.bias[i])

            print ("Terminé de calcular la salida de la capa actual**")

            
            self.activaciones.append(activacion)
        return self.activaciones[-1]

    def backpropagation(self, X, y):
        self.deltas = []
        salida = self.activaciones[-1]
        error = salida - y
        delta = error * self.activacion_derivada(salida)

        self.deltas.append(delta)

        # Propagar el error hacia atrás a través de la red neuronal
        for i in reversed(range(self.num_capas - 2)):

            if self.funciones_activacion[i] == "r":
                delta = np.dot(self.deltas[0], self.pesos[i + 1].T) * self.activacion_derivada_relu(self.activaciones[i + 1])
            elif self.funciones_activacion[i] == "t":
                delta = np.dot(self.deltas[0], self.pesos[i + 1].T) * self.activacion_derivada_tanh(self.activaciones[i + 1])
            else:
                delta = np.dot(self.deltas[0], self.pesos[i + 1].T) * self.activacion_derivada(self.activaciones[i + 1])
            
            self.deltas.insert(0, delta)

        # Actualizar pesos y bias
        for i in range(self.num_capas - 1):
            d_peso = np.outer(self.activaciones[i], self.deltas[i])
            self.pesos[i] -= self.alpha * d_peso
            self.bias[i] -= self.alpha * self.deltas[i]

    def fit(self, X, y):
        tiempo_inicio = time.time()
        for _ in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                self.feedforward(x)
                self.backpropagation(x, y_true)
        self.tiempo_entrenamiento = time.time() - tiempo_inicio

    def predict(self, X):
        self.activaciones = [X]
        for i in range(self.num_capas - 1):
            # Calcular la salida de la capa actual
            activacion = self.activacion(np.dot(self.activaciones[i], self.pesos[i]) + self.bias[i])
            self.activaciones.append(activacion)
        predicciones = np.argmax(self.activaciones[-1], axis=1)
        return predicciones
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def metricas(self, y_prueba, y_pred):
        recall = recall_score(y_prueba, y_pred, average='macro')
        precision = precision_score(y_prueba, y_pred, zero_division=1, average='macro')
        accuracy = accuracy_score(y_prueba, y_pred)
        f1 = f1_score(y_prueba, y_pred, average='macro')
        return recall, precision, accuracy, f1, self.tiempo_entrenamiento
    
    def imprimir_metricas(self, y_prueba, y_pred):
        recall, precision, accuracy, f1, tiempo_entrenamiento = self.metricas(y_prueba, y_pred)
        print ("Recall_______________________: ", recall)
        print ("Precision____________________: ", precision)
        print ("Accuracy_____________________: ", accuracy)
        print ("F1___________________________: ", f1)
        print ("Tiempo de entrenamiento______: ", tiempo_entrenamiento)

def myMLP (capas_ocultas, funciones_activacion, alpha, epochs):
    capas = [784] + capas_ocultas + [10]
    funciones_activacion = ["s"] + funciones_activacion + ["s"]
    perceptron = PerceptronMulticapa(capas, funciones_activacion, alpha, epochs)
    num_capas = perceptron.num_capas
    return perceptron

# Cargar y preparar los datos de MNIST
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data.astype('float32')
y = mnist.target.astype('int32')
X /= 255.0
print("Etiquetas únicas:", np.unique(y))
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el perceptrón multicapa

capas_ocultas = [8, 10, 61, 30]
funciones_activacion = ["s", "r", "t", "s"]
alpha = 0.1
epochs = 2
perceptron = myMLP(capas_ocultas, funciones_activacion, alpha, epochs)

# Crear y entrenar el perceptrón multicapa
entrada_dim = X_entrenamiento.shape[1]
#perceptron = PerceptronMulticapa(capas=[entrada_dim, 10, 10], alpha=0.1, epochs=5)

X_entrenamiento = X_entrenamiento.to_numpy()

perceptron.fit(X_entrenamiento, np.eye(10)[y_entrenamiento])

# Hacer predicciones sobre el conjunto de prueba
predicciones = perceptron.predict(X_prueba)

# Imprimir métricas
perceptron.imprimir_metricas(y_prueba, predicciones)