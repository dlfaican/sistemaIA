import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import numpy as np
import io

st.header('Universidad de las Fuerzas Armadas - ESPE', divider='rainbow')
st.subheader("Sistema de IA para optimizar la cadena de suministro")
st.caption("Domenica Faican Camacho")
st.divider()

# Cargar el conjunto de datos
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')  # Reemplazar con la ruta correcta

# Funciones de Optimizacion

def recopilar_procesar_datos():
    # Manejar valores faltantes
    df.dropna(subset=['Sales per customer'], inplace=True)
    df.fillna({'Benefit per order': 0}, inplace=True)

    # Convertir tipos de datos
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])

    # Otras operaciones 
    df['Customer Zipcode'] = df['Customer Zipcode'].astype(str)
    df['Order Status'] = df['Order Status'].str.upper()

    # Mostrar información actualizada del DataFrame
    st.write("Datos posterior al procesamiento y limpieza de las variables: ")
    # Captura la salida estándar en un objeto en memoria
    buffer = io.StringIO()
    df.info(buf=buffer)

    # Mostrar la información en Streamlit
    st.text(buffer.getvalue())

def implementar_algoritmos_aprendizaje():
    # Dividir el conjunto de datos en características (X) y variable objetivo (y)
    X = df[['Sales per customer']]
    y = df['Order Item Total']

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo de regresión lineal
    modelo_regresion = LinearRegression()

    # Entrenar el modelo con el conjunto de entrenamiento
    modelo_regresion.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = modelo_regresion.predict(X_test)

    # Evaluar el rendimiento del modelo
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Error cuadrático medio (MSE): {mse}")

    # Visualizar las predicciones
    plt.scatter(X_test, y_test, color='black', label='Datos reales')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicciones')
    plt.xlabel('Ventas por cliente')
    plt.ylabel('Total de ítems del pedido')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def analisis_exploratorio_datos():
    # Realizar un análisis exploratorio más completo, incluyendo visualizaciones y estadísticas descriptivas
    print("Estadísticas descriptivas del DataFrame:")
    st.write(df.describe())

    # Visualización de la distribución de ventas por cliente
    plt.figure(figsize=(10, 6))
    plt.hist(df['Sales per customer'], bins=30, color='blue', alpha=0.7)
    plt.title('Distribución de Ventas por Cliente')
    plt.xlabel('Ventas por Cliente')
    plt.ylabel('Frecuencia')
    st.pyplot()

def clustering():
    # Aplicar técnicas de clustering para segmentar datos
    features = df[['Days for shipping (real)', 'Sales per customer']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Utilizar KMeans para el clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Visualizar los clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Days for shipping (real)'], df['Sales per customer'], c=df['Cluster'], cmap='viridis')
    plt.title('Clustering de Datos de Envío y Ventas')
    plt.xlabel('Días de envío (real)')
    plt.ylabel('Ventas por Cliente')
    st.pyplot()

def disenar_arquitectura_sistema(data):
    # Crear un grafo dirigido
    G = nx.DiGraph()

    # Agregar nodos y aristas al grafo
    for _, row in data.iterrows():
        # Verificar si el nodo tiene información de posición
        if 'Longitude' in row and 'Latitude' in row:
            G.add_node(row['Order Id'], pos=(row['Longitude'], row['Latitude']))
            G.add_edge(row['Order Id'], row['Product Card Id'], weight=row['Days for shipping (real)'])

    return G

def integrar_algoritmos_optimizacion(G):
    # Verificar si el grafo tiene nodos
    if not G.nodes:
        print("El grafo no tiene nodos. Asegúrate de que los nodos y las aristas se agreguen correctamente.")
        return

    # Obtener el primer nodo del grafo
    start = next(iter(G.nodes))

    # Verificar si el grafo tiene aristas
    if not G.edges:
        print("El grafo no tiene aristas. Asegúrate de que los nodos y las aristas se agreguen correctamente.")
        return

    # Obtener el último nodo del grafo
    end = list(G.neighbors(start))[-1]

    # Calcular el camino más corto utilizando el algoritmo de Dijkstra
    shortest_path = nx.single_source_dijkstra(G, source=start, target=end)

    # Imprimir el camino más corto y sus tiempos
    st.write("Camino más corto:", shortest_path[1])
    st.write("Tiempo total:", shortest_path[0])

    # Verificar si los nodos en el camino más corto tienen información de posición
    nodes_with_position = [node for node in shortest_path[1] if G.nodes[node].get('pos') is not None]

    if nodes_with_position:
        # Visualizar el grafo con el camino más corto
        pos = nx.get_node_attributes(G, 'pos')
        nodes_to_draw = [node for node in shortest_path[1] if node in pos]
        subgraph = G.subgraph(nodes_to_draw)
        nx.draw(subgraph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels={(u, v): d['weight'] for u, v, d in subgraph.edges(data=True)})
        st.pyplot()
    else:
        st.write("Algunos nodos en el camino más corto no tienen información de posición y no se pueden visualizar.")

def calcular_huella_carbono(df):
    # Factores de emisión 
    factor_emision_transport = 0.202  # kg CO2 eq/km
    factor_emision_empaque = 0.1  # kg CO2 eq por unidad de empaque

    # Calcular la huella de carbono para actividades de transporte y empaque antes de la optimización
    df['Huella_Carbono_Transporte_Antes'] = df['Days for shipping (real)'] * factor_emision_transport
    df['Huella_Carbono_Empaque_Antes'] = df['Order Item Quantity'] * factor_emision_empaque
    df['Huella_Carbono_Total_Antes'] = df['Huella_Carbono_Transporte_Antes'] + df['Huella_Carbono_Empaque_Antes']

    # Visualización del gráfico de barras comparando huellas de carbono antes de la optimización
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df['Huella_Carbono_Total_Antes'], color='skyblue', label='Antes de la Optimización')
    plt.title('Huella de Carbono por Actividad (Antes de la Optimización)')
    plt.xlabel('Actividad')
    plt.ylabel('Huella de Carbono (kg CO2 eq)')
    plt.legend()
    st.pyplot()

    # Actualizar datos después de la optimización 
    df['Huella_Carbono_Transporte_Optimizado'] = df['Days for shipping (real)'] * 0.1  
    df['Huella_Carbono_Empaque_Optimizado'] = df['Order Item Quantity'] * factor_emision_empaque
    df['Huella_Carbono_Total_Optimizado'] = df['Huella_Carbono_Transporte_Optimizado'] + df['Huella_Carbono_Empaque_Optimizado']

    # Visualización del gráfico de barras comparando huellas de carbono después de la optimización
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df['Huella_Carbono_Total_Optimizado'], color='green', label='Después de la Optimización')
    plt.title('Huella de Carbono por Actividad (Después de la Optimización)')
    plt.xlabel('Actividad')
    plt.ylabel('Huella de Carbono (kg CO2 eq)')
    plt.legend()
    st.pyplot()

    # Mostrar la reducción en la huella de carbono
    reduccion_total = (df['Huella_Carbono_Total_Antes'] - df['Huella_Carbono_Total_Optimizado']).sum()
    st.write(f"Reducción total en la huella de carbono: {reduccion_total:.2f} kg CO2 eq")

def establecer_kpis():
    # Establecer KPIs relevantes para la cadena de suministro
    df['Profit per Order'] = df['Order Profit Per Order'] / df['Order Item Quantity']
    kpi_df = df[['Sales per customer', 'Profit per Order']]
    st.write("KPIs:")
    st.write(kpi_df.head())


def analizar_comparar_datos():
    # Analizar y comparar datos actuales con resultados previos
    sales_period_1 = df[df['order date (DateOrders)'] < '2018-06-01']['Sales per customer'].sum()
    sales_period_2 = df[df['order date (DateOrders)'] >= '2018-06-01']['Sales per customer'].sum()
    st.write(f"Ventas en el primer período: {sales_period_1}")
    st.write(f"Ventas en el segundo período: {sales_period_2}")

# Opciones del menu
menu_options = """
1. Recopilar y procesar datos
2. Implementar algoritmos de aprendizaje
3. Análisis exploratorio de datos
4. Clustering
5. Integrar algoritmos de optimización
6. Calcular huella de carbono
7. Establecer KPIs
"""

chosen_option = st.sidebar.selectbox("Elige una opción", menu_options.split("\n"))

# Execute the selected function
if chosen_option == "1. Recopilar y procesar datos":
    recopilar_procesar_datos()
elif chosen_option == "2. Implementar algoritmos de aprendizaje":
    implementar_algoritmos_aprendizaje()
elif chosen_option == "3. Análisis exploratorio de datos":
    analisis_exploratorio_datos()
elif chosen_option == "4. Clustering":
    clustering()
elif chosen_option == "5. Integrar algoritmos de optimización":
    integrar_algoritmos_optimizacion(disenar_arquitectura_sistema(df))
elif chosen_option == "6. Calcular huella de carbono":
    calcular_huella_carbono(df.copy())
elif chosen_option == "7. Establecer KPIs":
    establecer_kpis() 
    analizar_comparar_datos()

