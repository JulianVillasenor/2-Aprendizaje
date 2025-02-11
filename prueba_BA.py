import utileria as ut
import arboles_numericos as an
import bosque_aleatorio as ba
import os
import random

# Descarga y descomprime los datos
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
archivo = "datos/cancer.zip"
archivo_datos = "datos/wdbc.data"
atributos = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

# Extrae datos y convierte a numéricos
datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)
for d in datos:
    d['Diagnosis'] = 1 if d['Diagnosis'] == 'M' else 0
    for i in range(1, 31):
        d[f'feature_{i}'] = float(d[f'feature_{i}'])
    del(d['ID'])

# Selecciona el atributo objetivo
target = 'Diagnosis'

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Para diferentes números de árboles (M) y profundidades máximas
errores = []
for M in [10, 20, 30]:  # Número de árboles en el bosque
    for profundidad in [1, 3, 5, 10, 15, 20, 30]:  # Profundidad máxima de los árboles
        bosque = ba.entrena_bosque(
            datos_entrenamiento,
            target,
            clase_default=0,  # Clase predeterminada
            M=M,
            max_profundidad=profundidad
        )
        error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
        error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)
        errores.append((M, profundidad, error_en_muestra, error_en_validacion))

# Muestra los errores
print('M'.center(10) + 'd'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 50)
for M, profundidad, error_entrenamiento, error_validacion in errores:
    print(
        f'{M}'.center(10)
        + f'{profundidad}'.center(10)
        + f'{error_entrenamiento:.2f}'.center(15)
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 50 + '\n')

# Entrena con la mejor combinación de M y profundidad
mejor_M, mejor_profundidad = 20, 5  # Valores seleccionados basados en los resultados
bosque_final = ba.entrena_bosque(
    datos,
    target,
    clase_default=0,
    M=mejor_M,
    max_profundidad=mejor_profundidad
)

# Evalúa el bosque final
error_final = ba.evalua_bosque(bosque_final, datos, target)
print(f'Error del bosque final entrenado con TODOS los datos: {error_final:.2f}')

# Imprime algunos árboles del bosque
for i, arbol in enumerate(bosque_final[:3]):  # Imprime los primeros 3 árboles
    print(f"\nÁrbol {i + 1}:")
    an.imprime_arbol(arbol)