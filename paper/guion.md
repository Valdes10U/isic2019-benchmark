# Guion de la exposición

Revisión individual del jueves 15 de octubre: 5 a 6 minutos de exposición, 40 a 45 segundos por slide. Las slides de respaldo (9 a 16) solo se abren si las piden en las preguntas.

## 1. Título

Mi proyecto compara tres arquitecturas para clasificar lesiones de piel en ISIC 2019: ResNet50, EfficientNet-B0 y Swin-T. La pregunta no fue cuál llega al estado del arte, sino cómo se ordenan cuando les doy exactamente el mismo presupuesto. Las cifras publicadas sobre este dataset van de 74 a casi 92, pero cada trabajo usa su propia partición, resolución y métrica, y esas cifras no se pueden comparar entre sí. Por eso fijé el protocolo antes de entrenar y lo apliqué igual a los tres modelos.

## 2. Protocolo

El dataset tiene 25,331 imágenes en ocho clases, con 54 nevos por cada dermatofibroma. Partí los datos 70, 15 y 15 agrupando por lesión, porque ISIC tiene varias fotos de la misma lesión y, si caen en entrenamiento y en test, las métricas se inflan. Como evaluación externa usé el test oficial, 6,191 imágenes, que nunca usé para decidir nada. Los tres modelos se entrenan igual: 224 píxeles, AdamW, 15 épocas con early stopping, pérdida ponderada y tres semillas. Todo corrió en una RTX 5070 Ti de laptop.

## 3. Resultado principal

Swin-T gana en todas las métricas y en los dos tests. En F1 macro saca 58.9, contra 53.2 de EfficientNet y 52.0 de ResNet50. Esos 5.7 puntos son casi cuatro veces la desviación más alta entre semillas, así que no dependen de la semilla. En el test oficial el orden se mantiene. Entre las dos CNN el orden depende de la métrica: EfficientNet gana en F1 y ResNet50 en exactitud balanceada. La gráfica adelanta el costo: Swin-T es el más pesado y el más lento.

## 4. Costo y convergencia

En costo la historia cambia. EfficientNet-B0 necesita la décima parte del cómputo de los otros dos, 0.4 GMACs, y procesa más de 3 mil imágenes por segundo. Swin-T rinde más por época: en la quinta ya supera lo mejor que ResNet50 logra en quince, pero cada época le cuesta 46 segundos contra 25. ResNet50 queda dominado: EfficientNet lo supera en F1 y en AUROC con menos de todo. Y está subajustado: su F1 en entrenamiento es 72, contra 90 y 95. Sospecho de la tasa de aprendizaje compartida, aunque no lo probé.

## 5. Generalización y ensembles

Del test interno al oficial, el F1 macro cae 0.4 puntos o menos, menos que el ruido entre semillas. Las otras métricas sí caen, entre 1.7 y 4.3 puntos, así que el F1 solo no cuenta toda la historia. En los ensembles, mezclar las tres arquitecturas no supera a promediar tres semillas de Swin-T: 59.3 contra 60.5. Las CNN diluyen al mejor modelo. Tres EfficientNet promediados llegan a 56.7 con 1.2 GMACs, arriba de un ResNet50 con 4.1.

## 6. Ablación de la ponderación

Al ver un F1 macro bajo, escribí una hipótesis antes de correr nada: que ponderar por frecuencia inversa hunde la precisión de las clases raras y deprime el F1. Quitar los pesos tendría que subirlo, y bajó en los tres modelos. Lo que sí cambia es la exactitud balanceada. En ResNet50, sin pesos la exactitud sube a 72.7, pero la exactitud balanceada se desploma de 58.4 a 48.7: el modelo deja de predecir las clases raras, y en dermatofibroma el recall cae a 16.7. La hipótesis quedó descartada.

## 7. Interpretabilidad

Usé Grad-CAM sobre las mismas 16 imágenes en los tres modelos; aquí muestro cuatro. En las dos CNN los mapas caen sobre la lesión en la mayoría de los casos, y se pueden comparar entre sí. En Swin-T salen bloques en cruz que siguen las ventanas de siete por siete. En el nevo acierta con un mapa que cubre casi toda la imagen, y en la queratosis benigna el mapa sale vacío, porque el ReLU anula un gradiente negativo. Lo leo como una limitación del método: Grad-CAM se diseñó para capas convolucionales, y en Swin-T la capa objetivo es una normalización.

## 8. Conclusiones

La recomendación depende del costo. Si el cómputo no limita, Swin-T, y su ventaja se sostiene en el test oficial. Si hay que correr en hardware modesto, EfficientNet-B0, y promediar tres semillas cierra buena parte de la distancia. Con este protocolo, ResNet50 no conviene en ningún escenario. Las limitaciones: el nivel queda bajo la literatura por los 224 píxeles y las 15 épocas, un protocolo común no es el óptimo de cada arquitectura, y hay clases con 36 imágenes de test. Lo siguiente es ajustar la tasa por arquitectura, subir la resolución y probar attention rollout. Gracias.
