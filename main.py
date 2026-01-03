from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import httpx
from models import AccidenteRequest, AccidenteResponse, CombinedResponse
from utils import procesar_csv, combinar_datos
import io
import os
from pathlib import Path
from datetime import datetime
import shutil

app = FastAPI(title="Sistema de Análisis de Accidentes")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_BASE_URL = "http://localhost:8080/api/accidentes/cercanos"
CSV_DIRECTORY = "data"  # Directorio donde se guardarán los CSV

# Crear directorio si no existe
Path(CSV_DIRECTORY).mkdir(exist_ok=True)

def buscar_csv_defecto():
    """
    Busca el archivo CSV más reciente en el directorio de datos
    Retorna el path del CSV más reciente o None
    """
    csv_dir = Path(CSV_DIRECTORY)
    csv_files = list(csv_dir.glob("*.csv"))
    
    if csv_files:
        # Ordenar por fecha de modificación (más reciente primero)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return csv_files[0]
    return None

def listar_todos_csv():
    """
    Lista todos los archivos CSV en el directorio de datos
    """
    csv_dir = Path(CSV_DIRECTORY)
    csv_files = list(csv_dir.glob("*.csv"))
    
    archivos = []
    for csv_file in csv_files:
        stat = csv_file.stat()
        archivos.append({
            "nombre": csv_file.name,
            "path": str(csv_file),
            "tamaño_bytes": stat.st_size,
            "fecha_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "es_mas_reciente": csv_file == buscar_csv_defecto()
        })
    
    # Ordenar por fecha de modificación (más reciente primero)
    archivos.sort(key=lambda x: x["fecha_modificacion"], reverse=True)
    return archivos

async def guardar_csv(archivo: UploadFile) -> Path:
    """
    Guarda el archivo CSV subido en el directorio de datos
    Retorna el path del archivo guardado
    """
    # Generar nombre único con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_original = archivo.filename.replace('.csv', '')
    nombre_archivo = f"{nombre_original}_{timestamp}.csv"
    
    # Path completo
    file_path = Path(CSV_DIRECTORY) / nombre_archivo
    
    # Guardar el archivo
    contenido = await archivo.read()
    with open(file_path, 'wb') as f:
        f.write(contenido)
    
    print(f"💾 CSV guardado: {nombre_archivo} ({len(contenido)} bytes)")
    
    return file_path

@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Accidentes de Tránsito",
        "version": "1.0",
        "endpoints": {
            "analizar": "POST /analizar",
            "estadisticas": "POST /estadisticas-csv",
            "test_api": "GET /api-externa/test",
            "csv_disponible": "GET /csv-disponible",
            "listar_csv": "GET /listar-csv",
            "eliminar_csv": "DELETE /eliminar-csv/{nombre}"
        }
    }

@app.get("/csv-disponible")
async def verificar_csv_disponible():
    """
    Verifica si existe un CSV por defecto (el más reciente) en el directorio de datos
    """
    csv_path = buscar_csv_defecto()
    
    if csv_path:
        stat = csv_path.stat()
        return {
            "existe": True,
            "nombre": csv_path.name,
            "path": str(csv_path),
            "tamaño_bytes": stat.st_size,
            "fecha_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    return {
        "existe": False,
        "mensaje": f"No se encontró ningún archivo CSV en el directorio '{CSV_DIRECTORY}'"
    }

@app.get("/listar-csv")
async def listar_csv_guardados():
    """
    Lista todos los archivos CSV guardados en el servidor
    """
    archivos = listar_todos_csv()
    
    return {
        "total": len(archivos),
        "archivos": archivos,
        "directorio": CSV_DIRECTORY
    }

@app.delete("/eliminar-csv/{nombre_archivo}")
async def eliminar_csv(nombre_archivo: str):
    """
    Elimina un archivo CSV específico del servidor
    """
    try:
        file_path = Path(CSV_DIRECTORY) / nombre_archivo
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"El archivo '{nombre_archivo}' no existe"
            )
        
        file_path.unlink()
        print(f"🗑️ CSV eliminado: {nombre_archivo}")
        
        return {
            "success": True,
            "mensaje": f"Archivo '{nombre_archivo}' eliminado correctamente",
            "archivos_restantes": len(listar_todos_csv())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar archivo: {str(e)}")

@app.post("/analizar")
async def analizar_accidentes(
    archivo: Optional[UploadFile] = File(None),
    usar_csv_defecto: Optional[str] = Form(None),
    latitud: float = Form(default=-2.89264),
    longitud: float = Form(default=-78.77814),
    radio_km: float = Form(default=5.0)
):
    """
    Analiza accidentes combinando datos del CSV y la API externa
    Si se sube un archivo nuevo, se guarda automáticamente y se convierte en el por defecto
    
    Args:
        archivo: Archivo CSV (opcional si se usa CSV por defecto)
        usar_csv_defecto: "true" para usar el CSV por defecto del servidor
        latitud: Latitud del punto de referencia
        longitud: Longitud del punto de referencia
        radio_km: Radio de búsqueda en kilómetros
    """
    try:
        df_csv = None
        archivo_usado = None
        
        # Determinar qué CSV usar
        if usar_csv_defecto == "true":
            # Usar CSV por defecto (el más reciente)
            csv_path = buscar_csv_defecto()
            
            if not csv_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró ningún archivo CSV en el directorio '{CSV_DIRECTORY}'. Por favor, carga un archivo CSV."
                )
            
            print(f"📂 Usando CSV por defecto: {csv_path.name}")
            df_csv = pd.read_csv(csv_path)
            archivo_usado = csv_path.name
            
        elif archivo:
            # Validar que sea CSV
            if not archivo.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=400, 
                    detail="El archivo debe ser un CSV"
                )
            
            # Guardar el archivo en el servidor
            saved_path = await guardar_csv(archivo)
            
            # Leer el archivo guardado
            df_csv = pd.read_csv(saved_path)
            archivo_usado = saved_path.name
            
            print(f"📤 CSV nuevo guardado y en uso: {archivo_usado}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Debes proporcionar un archivo CSV o usar el CSV por defecto"
            )
        
        # Validar que el CSV tenga datos
        if len(df_csv) == 0:
            raise HTTPException(
                status_code=400,
                detail="El CSV no contiene datos"
            )
        
        # Verificar columnas requeridas
        required_cols = ['latitud', 'longitud', 'tipo_accidente', 'provincia', 'ciudad']
        missing_cols = [col for col in required_cols if col not in df_csv.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {', '.join(missing_cols)}"
            )
        
        datos_csv = procesar_csv(df_csv, latitud, longitud, radio_km)
        
        # 2. Consumir la API externa
        datos_api = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    API_BASE_URL,
                    params={"lat": latitud, "lon": longitud}
                )
                
                if response.status_code == 200:
                    datos_api = response.json()
                    print(f"🌐 API externa retornó {len(datos_api)} accidentes")
                else:
                    print(f"⚠️ API externa retornó código {response.status_code}")
                    
        except httpx.ConnectError:
            print("⚠️ No se pudo conectar a la API externa (continuando sin datos de API)")
        except httpx.TimeoutException:
            print("⚠️ Timeout al conectar con API externa (continuando sin datos de API)")
        except Exception as e:
            print(f"⚠️ Error al consumir API externa: {str(e)} (continuando sin datos de API)")
        
        # 3. Combinar y analizar datos
        resultado = combinar_datos(datos_csv, datos_api, latitud, longitud)
        
        # Agregar información del archivo usado
        resultado["archivo_csv_usado"] = archivo_usado
        resultado["total_archivos_guardados"] = len(listar_todos_csv())
        
        return resultado
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error al parsear CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")

@app.post("/estadisticas-csv")
async def estadisticas_csv(archivo: UploadFile = File(...)):
    """
    Obtiene estadísticas básicas del CSV cargado
    """
    try:
        contenido = await archivo.read()
        df = pd.read_csv(io.BytesIO(contenido))
        
        return {
            "total_registros": len(df),
            "columnas": list(df.columns),
            "provincias": df['provincia'].value_counts().to_dict() if 'provincia' in df.columns else {},
            "ciudades": df['ciudad'].value_counts().head(10).to_dict() if 'ciudad' in df.columns else {},
            "tipos_accidente": df['tipo_accidente'].value_counts().to_dict() if 'tipo_accidente' in df.columns else {},
            "fechas": {
                "min": str(df['fecha'].min()) if 'fecha' in df.columns else None,
                "max": str(df['fecha'].max()) if 'fecha' in df.columns else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api-externa/test")
async def test_api_externa(lat: float = -2.89264, lon: float = -78.77814):
    """
    Prueba la conexión con la API externa
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                API_BASE_URL,
                params={"lat": lat, "lon": lon}
            )
            response.raise_for_status()
            data = response.json()
            return {
                "status": "success",
                "total_accidentes": len(data) if isinstance(data, list) else 0,
                "data": data
            }
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502, 
            detail="No se pudo conectar a la API externa. Verifica que esté corriendo en http://localhost:8080"
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout al conectar con API externa")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Verifica el estado de la API
    """
    csv_disponible = buscar_csv_defecto()
    total_csv = len(listar_todos_csv())
    
    return {
        "status": "healthy",
        "service": "Sistema de Análisis de Accidentes",
        "version": "1.0",
        "csv_por_defecto": {
            "disponible": csv_disponible is not None,
            "archivo": csv_disponible.name if csv_disponible else None
        },
        "total_archivos_csv": total_csv
    }