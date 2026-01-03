import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from math import radians, cos, sin, asin, sqrt
from collections import Counter
import os

def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula de Haversine
    Retorna la distancia en kilómetros
    """
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Fórmula de Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radio de la Tierra en km
    r = 6371
    
    return c * r

def procesar_csv(df: pd.DataFrame, lat_ref: float, lon_ref: float, radio_km: float) -> pd.DataFrame:
    """
    Procesa el DataFrame del CSV y calcula distancias desde el punto de referencia
    """
    # Calcular distancia para cada registro
    df['distancia_km'] = df.apply(
        lambda row: calcular_distancia_haversine(
            lat_ref, lon_ref, row['latitud'], row['longitud']
        ),
        axis=1
    )
    
    # Filtrar por radio
    df_filtrado = df[df['distancia_km'] <= radio_km].copy()
    
    # Ordenar por distancia
    df_filtrado = df_filtrado.sort_values('distancia_km')
    
    return df_filtrado

def identificar_zonas_peligrosas(accidentes: List[Tuple[float, float]], radio_metros: float = 500) -> List[Dict]:
    """
    Identifica zonas con alta concentración de accidentes
    """
    if not accidentes:
        return []
    
    zonas = []
    procesados = set()
    
    for i, (lat1, lon1) in enumerate(accidentes):
        if i in procesados:
            continue
            
        # Contar accidentes cercanos
        cluster = []
        for j, (lat2, lon2) in enumerate(accidentes):
            distancia = calcular_distancia_haversine(lat1, lon1, lat2, lon2) * 1000  # a metros
            if distancia <= radio_metros:
                cluster.append(j)
                procesados.add(j)
        
        if len(cluster) >= 3:  # Zona peligrosa si hay 3+ accidentes
            # Calcular centroide
            lats = [accidentes[j][0] for j in cluster]
            lons = [accidentes[j][1] for j in cluster]
            
            nivel = "ALTO" if len(cluster) >= 10 else "MEDIO" if len(cluster) >= 5 else "BAJO"
            
            zonas.append({
                "latitud": np.mean(lats),
                "longitud": np.mean(lons),
                "cantidad_accidentes": len(cluster),
                "radio_metros": radio_metros,
                "nivel_peligro": nivel
            })
    
    return sorted(zonas, key=lambda x: x['cantidad_accidentes'], reverse=True)

def generar_recomendaciones(estadisticas: Dict, zonas_peligrosas: List[Dict]) -> List[str]:
    """
    Genera recomendaciones basadas en el análisis
    """
    recomendaciones = []
    
    total = estadisticas['total_combinado']
    
    if total > 50:
        recomendaciones.append("⚠️ Zona de alta accidentalidad. Se recomienda extremar precauciones.")
    
    if len(zonas_peligrosas) > 0:
        recomendaciones.append(f"🔴 Se identificaron {len(zonas_peligrosas)} zonas de alta peligrosidad.")
        recomendaciones.append("📍 Evite las zonas marcadas en rojo cuando sea posible.")
    
    tipos_comunes = estadisticas.get('tipos_mas_comunes', {})
    if tipos_comunes:
        tipo_principal = max(tipos_comunes, key=tipos_comunes.get)
        recomendaciones.append(f"⚡ Tipo de accidente más común: {tipo_principal}. Manténgase alerta.")
    
    if estadisticas['total_csv'] > estadisticas['total_api']:
        recomendaciones.append("📊 Los datos históricos muestran mayor accidentalidad que reportes recientes.")
    
    recomendaciones.append("🚗 Respete los límites de velocidad y mantenga distancia de seguridad.")
    recomendaciones.append("🌧️ En clima 'Cubierto' (alta humedad), reduzca velocidad por posible lluvia.")
    
    return recomendaciones

def combinar_datos(df_csv: pd.DataFrame, datos_api: List[Dict], lat_ref: float, lon_ref: float) -> Dict:
    """
    Combina datos del CSV y la API para generar análisis completo
    """
    # Procesar datos API
    accidentes_api = []
    for acc in datos_api:
        distancia = calcular_distancia_haversine(
            lat_ref, lon_ref, acc['latitud'], acc['longitud']
        )
        acc['distancia_km'] = round(distancia, 2)
        accidentes_api.append(acc)
    
    # Convertir CSV a lista de diccionarios
    accidentes_csv = df_csv.to_dict('records')
    
    # Recolectar todos los puntos para análisis de zonas
    todos_puntos = [(row['latitud'], row['longitud']) for row in accidentes_csv]
    todos_puntos.extend([(acc['latitud'], acc['longitud']) for acc in accidentes_api])
    
    # Identificar zonas peligrosas
    zonas = identificar_zonas_peligrosas(todos_puntos)
    
    # Estadísticas separadas para CSV
    tipos_csv = Counter()
    provincias_csv = Counter()
    ciudades_csv = Counter()
    
    for row in accidentes_csv:
        if 'tipo_accidente' in row and row['tipo_accidente']:
            tipos_csv[str(row['tipo_accidente'])] += 1
        if 'provincia' in row and row['provincia']:
            provincias_csv[str(row['provincia'])] += 1
        if 'ciudad' in row and row['ciudad']:
            ciudades_csv[str(row['ciudad'])] += 1
    
    # Estadísticas separadas para API
    tipos_api = Counter()
    provincias_api = Counter()
    ciudades_api = Counter()
    
    for acc in accidentes_api:
        # Extraer tipo de accidente del modelo API
        if 'tipoaccidente' in acc and acc['tipoaccidente'] and isinstance(acc['tipoaccidente'], dict):
            tipo_nombre = acc['tipoaccidente'].get('nombre', None)
            if tipo_nombre:
                tipos_api[str(tipo_nombre)] += 1
        
        # Extraer provincia y ciudad del modelo API
        if 'ruta' in acc and acc['ruta'] and isinstance(acc['ruta'], dict):
            ruta = acc['ruta']
            
            if 'ciudad' in ruta and ruta['ciudad'] and isinstance(ruta['ciudad'], dict):
                ciudad_data = ruta['ciudad']
                
                # Extraer nombre de ciudad
                if 'nombreCiudad' in ciudad_data and ciudad_data['nombreCiudad']:
                    ciudades_api[str(ciudad_data['nombreCiudad'])] += 1
                
                # Extraer nombre de provincia
                if 'provincia' in ciudad_data and ciudad_data['provincia'] and isinstance(ciudad_data['provincia'], dict):
                    provincia_nombre = ciudad_data['provincia'].get('nombreProvincia', None)
                    if provincia_nombre:
                        provincias_api[str(provincia_nombre)] += 1
    
    # Combinar estadísticas
    tipos_combinados = tipos_csv + tipos_api
    provincias_combinadas = provincias_csv + provincias_api
    ciudades_combinadas = ciudades_csv + ciudades_api
    
    # Debug: imprimir estadísticas
    print(f"\n=== DEBUG ESTADÍSTICAS ===")
    print(f"Tipos CSV: {dict(tipos_csv)}")
    print(f"Tipos API: {dict(tipos_api)}")
    print(f"Provincias CSV: {dict(provincias_csv)}")
    print(f"Provincias API: {dict(provincias_api)}")
    print(f"Total accidentes API procesados: {len(accidentes_api)}")
    
    estadisticas = {
        "total_csv": len(accidentes_csv),
        "total_api": len(accidentes_api),
        "total_combinado": len(accidentes_csv) + len(accidentes_api),
        "accidentes_en_radio": len(df_csv),
        "tipos_mas_comunes": dict(tipos_combinados.most_common(10)),
        "provincias_afectadas": dict(provincias_combinadas),
        "ciudades_afectadas": dict(ciudades_combinadas.most_common(10)),
        # Estadísticas separadas
        "tipos_csv": dict(tipos_csv.most_common(10)) if tipos_csv else {},
        "tipos_api": dict(tipos_api.most_common(10)) if tipos_api else {},
        "provincias_csv": dict(provincias_csv) if provincias_csv else {},
        "provincias_api": dict(provincias_api) if provincias_api else {},
        "ciudades_csv": dict(ciudades_csv.most_common(10)) if ciudades_csv else {},
        "ciudades_api": dict(ciudades_api.most_common(10)) if ciudades_api else {}
    }
    
    recomendaciones = generar_recomendaciones(estadisticas, zonas)
    
    return {
        "punto_referencia": {"latitud": lat_ref, "longitud": lon_ref},
        "radio_busqueda_km": df_csv['distancia_km'].max() if len(df_csv) > 0 else 0,
        "estadisticas": estadisticas,
        "accidentes_csv": accidentes_csv,
        "accidentes_api": accidentes_api,
        "zonas_peligrosas": zonas,
        "recomendaciones": recomendaciones
    }