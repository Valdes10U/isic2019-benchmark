# Notas y posibles mejoras (no implementadas)

- F1 de train para la brecha de generalización: `history.csv` no lo incluye. Propuesta para la Fase 3: calcularlo en `evaluate.py` con el mejor checkpoint sobre train sin augmentation (transform de val).
- El resize directo a 256x256 deforma las imágenes de 600x450 (4:3), que son una fracción importante de train y test. Afecta igual a todos los modelos; se puede mencionar como limitación en el artículo.
