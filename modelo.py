import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam  
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

# Lista de clases
clasesArañas = [
    "Violinista",
    "Lobo",
    "Saltarina",
    "DeJardin",
    "ViudaNegra"
]

# Procesar imagen 
def preprocesarImagen(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(imagen, (64, 64))
    imagen = imagen.astype('float32') / 255.0
    return imagen

# Cargar imágenes y etiquetas por clase y rango
def cargarImagenesConEtiquetas(rutaBase, modo, rangoInicio, rangoFin):
    imagenes = []
    etiquetas = []

    for indiceClase, nombreClase in enumerate(clasesArañas):
        carpetaClase = os.path.join(rutaBase, modo, nombreClase)

        if not os.path.exists(carpetaClase):
            print(f"No se encontró la carpeta {carpetaClase}")
            continue

        for i in range(rangoInicio, rangoFin + 1):
            nombreArchivo = f"{nombreClase} ({i}).png"
            rutaImagen = os.path.join(carpetaClase, nombreArchivo)

            if not os.path.exists(rutaImagen):
                continue

            try:
                imagen = cv2.imread(rutaImagen)
                if imagen is None:
                    print(f"Error al leer {rutaImagen}")
                    continue

                imagenProcesada = preprocesarImagen(imagen)
                imagenes.append(imagenProcesada)
                etiquetas.append(indiceClase)
            except Exception as e:
                print(f"Error procesando {rutaImagen}: {str(e)}")

    return np.array(imagenes), np.array(etiquetas)


# Carga de datos
rutaBase = 'Spyder'
imagenesEntrenamiento, etiquetasEntrenamiento = cargarImagenesConEtiquetas(rutaBase, "train", 1, 133)
imagenesValidacion, etiquetasValidacion = cargarImagenesConEtiquetas(rutaBase, "evaluation", 134, 172)
imagenesPrueba, etiquetasPrueba = cargarImagenesConEtiquetas(rutaBase, "test", 173, 190)

# Modelo CNN
modeloSpyder = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(clasesArañas), activation='softmax')
])

modeloSpyder.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['sparse_categorical_accuracy']
)

# Entrenamiento del modelo
historialEntrenamiento= modeloSpyder.fit(
    imagenesEntrenamiento, etiquetasEntrenamiento,
    validation_data=(imagenesValidacion, etiquetasValidacion),
    epochs=150,
    verbose=1
)

modeloSpyder.save("modeloSpyder.h5")






# Evaluación
pérdidaPrueba, precisiónPrueba = modeloSpyder.evaluate(imagenesPrueba, etiquetasPrueba, verbose=0)
print(f"\nPrecisión en prueba: {precisiónPrueba:.2%}")

# Gráficos de rendimiento
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historialEntrenamiento.history['sparse_categorical_accuracy'], label='Precisión entrenamiento')
plt.plot(historialEntrenamiento.history['val_sparse_categorical_accuracy'], label='Precisión validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historialEntrenamiento.history['loss'], label='Error entrenamiento')
plt.plot(historialEntrenamiento.history['val_loss'], label='Error validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()


# Función para probar una imagen individual
def probarModeloConImagen(rutaImagen):
    if not os.path.exists(rutaImagen):
        print(f"No se encontró {rutaImagen}")
        return

    modelo = tf.keras.models.load_model("modeloSpyder.h5")
    
    try:
        imagen = cv2.imread(rutaImagen)
        if imagen is None:
            print("Error: No se pudo leer la imagen")
            return

        imagenProcesada = preprocesarImagen(imagen)
        imagenProcesadaExpandida = np.expand_dims(imagenProcesada, axis=0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title("Imagen original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(imagenProcesada, 0, 1))
        plt.title("Imagen procesada")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Predicción
        prediccion = modelo.predict(imagenProcesadaExpandida)
        clasePredicha = np.argmax(prediccion)
        confianza = np.max(prediccion)

        print("\nResultado:")
        print(f"Especie de araña: {clasesArañas[clasePredicha]}")
        print(f"Confianza: {confianza:.2%}")

    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")

# Probar con una imagen de ejemplo
probarModeloConImagen('Spyder/test/Lobo/Lobo (192).png')
