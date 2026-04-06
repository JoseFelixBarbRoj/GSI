Dataset url: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
Se recomienda **utilizar entornos virtuales** para instalar el proyecto.
Se asume instalación previa de Python 3.12+

El presente proyecto consiste en **3 sistemas**: uno de **entrenamiento** de 3 modelos diferentes para clasificación de imágenes del dataset cuya url se adjunta (**clasificación de mariposas según su especie**), otro de **testing** de esos modelos y finalmente un prototipo de aplicación que permitiría el **etiquetado automático de imágenes de mariposas, clasificándolas en hasta 75 especies.**

Para instalar:

1). Clonar el repo: `git clone https://github.com/JoseFelixBarbRoj/GSI.git`

2) Crear un entorno virtual con `python3 -m venv venv`

3) Activar el entonrno virtual:

    a) En Linux (recomendado): `source venv/bin/activate`

    b) En Windows: `.\venv\Scripts\activate`

4) Instalar las dependencias del proyecto en el entorno virtual:

    a) Para usuarios finales: `pip install .`

    b) Para desarrolladores del proyecto: `pip install -e .`


5) (Opcional) Para reentrenar un modelo: `python scripts/train_model.py <modelo> <num_epochs>`, con modelo en {BaselineModel, ExtendedBaselineModel, EfficientNetV2} y siendo num_epochs un entero (número de épocas para el entrenamiento). El script  sobreescribirá el modelo (archivo.pth) de  `models/<modelo>` y las gráficas de precisión y pérdida en el entrenamiento (`acc_curve.png` y `loss_curve.png` )

6) (Opcional) Para testear un modelo: `python scripts/test_model.py <modelo> models`, con modelo en {BaselineModel, ExtendedBaselineModel, EfficientNetV2}. El script generará la gráfica (testing.png) con accuracy, accuracy top3 y accuracy top5 del modelo correspondiente en `models/<modelo>`

7) Para probar el prototipo de aplicación: `python app.py <modelo> <entrada>`, con modelo en {BaselineModel, ExtendedBaselineModel, EfficientNetV2} y siendo entrada tanto un fichero con formato de imagen (.png, .jpg...) o un directorio de imágenes. Si es un solo archivo, el sistema imprime en su salida estándar la clase predicha por el modelo y la probabilidad asignada (grado de confianza). Si es un directorio, se genera un archivo .csv en la ruta indicada en la salida estándar, conteniendo las etiquetas asignadas y la probabilidad que el modelo da a cada asignación. **IMPORTANTE**: El dataset generado **ya contiene un directorio de imágenes sin etiquetar**. Por ello basta con ejecutar `python app.py <modelo> data/test/` (o extender la ruta si se quiere probar a etiquetar una sola, añadiéndole el nombre de esa imagen individual)