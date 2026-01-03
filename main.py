from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import httpx
from models import AccidenteRequest, AccidenteResponse, CombinedResponse
from utils import procesar_csv, combinar_datos
import io

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

@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Accidentes de Tránsito",
        "version": "1.0",
        "endpoints": {
            "analizar": "POST /analizar",
            "estadisticas": "POST /estadisticas-csv",
            "test_api": "GET /api-externa/test"
        }
    }

@app.post("/analizar")
async def analizar_accidentes(
    archivo: UploadFile = File(..., description="Archivo CSV con datos de accidentes"),
    latitud: float = Form(default=-2.89264),
    longitud: float = Form(default=-78.77814),
    radio_km: float = Form(default=5.0)
):
    """
    Analiza accidentes combinando datos del CSV cargado y la API externa
    
    Args:
        archivo: Archivo CSV
        latitud: Latitud del punto de referencia
        longitud: Longitud del punto de referencia
        radio_km: Radio de búsqueda en kilómetros
    """
    try:
        # Validar que sea un archivo CSV
        if not archivo.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe ser un CSV"
            )
        
        # 1. Leer y procesar el CSV
        contenido = await archivo.read()
        
        if len(contenido) == 0:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío"
            )
        
        df_csv = pd.read_csv(io.BytesIO(contenido))
        
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
    return {
        "status": "healthy",
        "service": "Sistema de Análisis de Accidentes",
        "version": "1.0"
    }