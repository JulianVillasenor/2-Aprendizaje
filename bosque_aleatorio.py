import random
from arboles_numericos import entrena_arbol, imprime_arbol, predice_arbol
from collections import Counter

def entrena_bosque(datos, target, clase_default, M=10, max_profundidad=None, acc_nodo=1.0, min_ejemplos=0, variables_por_nodo=None):
    bosque = [] #Lista de arboles entrenados
    n = len(datos)  
    random.seed(42)
    for _ in range(M):
        
        if variables_por_nodo  is None :
            variables_por_nodo = random.randint(1, len(datos[0].keys())-1)
        # bootstrapping
        subconjunto = [random.choice(datos) for _ in range(n)]
        
        # Entrenar un árbol con el subconjunto y agregarlo al bosque
        arbol = entrena_arbol(
            subconjunto, 
            target, 
            clase_default, 
            max_profundidad=max_profundidad, 
            acc_nodo=acc_nodo, 
            min_ejemplos=min_ejemplos, 
            variables_seleccionadas=variables_por_nodo
        )
        bosque.append(arbol)

    return bosque

def predice_bosque(bosque, instancia):
    # Obtener las predicciones de cada árbol en el bosque
    predicciones = [arbol.predice(instancia) for arbol in bosque]
    clase_predicha = Counter(predicciones).most_common(1)[0][0] #de todo el bosque devuelve a la mas comun
    return clase_predicha

def evalua_bosque(bosque, datos, target):
    if not bosque:
        raise ValueError("El bosque está vacío, no se puede evaluar.")

    if not datos:
        raise ValueError("No hay datos para evaluar el bosque.")

    if any(target not in d for d in datos):
        raise ValueError(f"El atributo '{target}' no está presente en todos los datos.")

    # Obtener las predicciones para todas las instancias
    predicciones = [predice_bosque(bosque, d) for d in datos]
    
    # Calcular la precisión
    precision = sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)
    
    return precision

def main():
    datos = [
        {"atributo1": 1, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 1, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 2, "clase": "positiva"},
        {"atributo1": 1, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 2, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 3, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 4, "atributo2": 3, "clase": "negativa"},
        {"atributo1": 1, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 2, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 3, "atributo2": 4, "clase": "positiva"},
        {"atributo1": 4, "atributo2": 4, "clase": "positiva"},
     
   ]
    bosque = entrena_bosque(datos, "clase", "positiva")

    for i, arbol in enumerate(bosque):
        print(f"\nÁrbol {i + 1}:")
        imprime_arbol(arbol)

    acc = evalua_bosque(bosque, datos, "clase")
    print(f"El acierto en los mismos datos que se entrenó es {acc}")

if __name__ == "__main__":
    main()