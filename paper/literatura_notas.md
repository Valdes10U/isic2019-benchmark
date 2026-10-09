# Notas de literatura para el artículo

Qué aporta cada referencia y en qué sección va. Las cifras que se citan aquí salen de los propios artículos (abstract o tablas), revisadas el 6 de octubre de 2026. Los números de distintos trabajos no son comparables entre sí: usan particiones, resoluciones, datos externos y métricas distintas. Eso mismo es un argumento para la introducción: la falta de un protocolo común es lo que motiva este trabajo.

## Introducción y motivación

- esteva2017dermatologist. El trabajo que puso la clasificación de lesiones cutáneas con CNNs en el mapa (Nature, 2017): una Inception v3 alcanza desempeño comparable al de dermatólogos en una tarea binaria. Sirve para abrir.
- tschandl2019comparison. Estudio internacional (Lancet Oncology, 2019) sobre el reto ISIC 2018: los algoritmos superaron en promedio a los lectores humanos, pero la ventaja se reduce fuera de la distribución de entrenamiento. Da la motivación clínica y adelanta el problema de generalización.
- daneshjou2022disparities. Modelos de dermatología de última generación caen fuerte en un conjunto externo con tonos de piel diversos y enfermedades poco comunes (Science Advances, 2022). Justifica evaluar en un conjunto con otra distribución (nuestro test oficial) y no solo en la partición interna.

## Estado del arte sobre ISIC 2019

- gessert2020isic. Ganador del reto ISIC 2019. Ensemble de EfficientNets multirresolución, SENet y ResNeXt, con pérdida balanceada, augmentation fuerte y metadata; reporta balanced accuracy de 74.2% en validación cruzada de cinco folds. Es la referencia de "lo que funciona con CNNs" en este dataset y de por qué los ensembles son el estándar en los retos. Nuestro ensemble simple se compara conceptualmente con esto.
- cassidy2022isic. Analiza las ediciones 2016 a 2020 de ISIC y encuentra una cantidad significativa de imágenes duplicadas dentro de cada edición y entre ediciones; propone una estrategia de limpieza y recomendaciones. Justifica directamente nuestra decisión de particionar agrupando por lesion_id: sin eso, imágenes de la misma lesión caen en train y test y las métricas se inflan.
- ayas2023swin. Swin Transformer para clasificación multiclase en ISIC 2019 (Neural Computing and Applications, 2023); reporta balanced accuracy de 82.3%. Evidencia de que los Transformers jerárquicos son competitivos en este dataset.
- aruk2026comparison. Lo más cercano a nuestro proyecto: compara 15 CNNs y 15 ViTs bajo las mismas condiciones en HAM10000 e ISIC 2019, incluyendo parámetros y costo computacional. Concluye que los modelos tipo Swin obtienen la mejor exactitud (0.9187 en ISIC 2019) pero con más parámetros y cómputo que las CNNs. Hay que posicionarnos frente a este trabajo: nosotros agregamos semillas múltiples, evaluación externa, Grad-CAM comparable y partición por lesión. Ojo: su exactitud es sobre una partición propia; no es comparable con la nuestra.

## CNN contra Transformer en imagen médica (marco general)

- matsoukas2021replace y matsoukas2023pretrained. Comparan ResNet50 contra DeiT-S bajo el mismo protocolo en tres datasets médicos, uno de ellos ISIC 2019. Hallazgos clave para nuestra discusión: desde cero las CNNs ganan por mucho; con pesos de ImageNet la brecha desaparece (en ISIC 2019 reportan recall de 0.810 para ResNet50 contra 0.844 para DeiT-S, mediana de 5 repeticiones); con preentrenamiento autosupervisado los ViTs ganan por poco. Además argumentan que los mapas de atención del Transformer son más finos que Grad-CAM en CNNs, lo que conecta con nuestra sección de interpretabilidad. Nota: ellos usan resize a 256 y Adam con lr 1e-4, muy parecido a nuestro protocolo; conviene decirlo.
- raghu2019transfusion. Explica por qué el transfer learning desde ImageNet ayuda en imagen médica aunque los dominios sean distintos (mejor inicialización y escala de pesos más que reutilización de features). Justifica usar pesos de ImageNet en los tres modelos.
- shamshad2023transformers. Survey de Transformers en imagen médica (Medical Image Analysis, 2023). Para situar el panorama y para la sección de trabajo futuro (autosupervisión, multimodalidad).

## Metodología (arquitecturas y herramientas)

- he2016resnet (ResNet50), tan2019efficientnet (EfficientNet-B0), liu2021swin (Swin-T), dosovitskiy2021vit (ViT, para explicar de dónde viene Swin).
- loshchilov2019adamw: AdamW.
- selvaraju2017gradcam: Grad-CAM.

## Datos (van con datasets.bib)

- tschandl2018ham10000, codella2017isic, hernandezperez2024bcn20000: citas obligatorias por licencia, ya en datasets.bib.
- combalia2019bcn20000 y codella2019isic2018: los preprints que ISIC enlaza en la atribución; se citan junto a los anteriores en la descripción del dataset.

## Qué falta y cómo se resuelve

- Cotejar DOI y páginas del bloque B de literatura.bib en la Fase 4 (Crossref o página del editor). Son trabajos muy conocidos, pero la regla es no compilar nada sin verificar.
- Completar nombres de pila de Aruk, Pacal y Toprak (2026) y de Ayas (2023); en el .bib van con iniciales porque las fuentes consultadas solo dan eso. No afecta el formato IEEE, que usa iniciales.
- Falta DOI de hernandezperez2024bcn20000 en datasets.bib; se busca en la Fase 4.
- Si en el artículo se quiere una tabla de "trabajos previos en ISIC 2019" con sus métricas, usar solo gessert2020isic, ayas2023swin, aruk2026comparison y matsoukas2021replace, y aclarar en el pie que las particiones difieren.
