import os
import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field # Pydantic V2 style
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from openai import AzureOpenAI

# --- App FastAPI ---
app = FastAPI(title="results_fetcher",)

# --- Configuración Inicial ---
load_dotenv()

# --- Elegir el modelo a usar ---
# MODEL = "gemini" 
# MODEL = "deepseek" 
MODEL = "openai" 

if MODEL == "gemini":
    logging.info("Usando modelo Gemini para la generación de contenido.")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logging.error("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logging.info("Cliente de Gemini configurado correctamente.")
        except Exception as e:
            logging.error(f"Error configurando el cliente de Gemini: {e}")
    GEMINI_MODEL_NAME = "gemini-2.0-flash" # Use a valid model
elif MODEL == "deepseek":
    logging.info("Usando modelo DeepSeek para la generación de contenido.")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        logging.error("Error: La variable de entorno DEEPSEEK_API_KEY no está configurada.")
    else:
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            logging.info("Cliente de DeepSeek configurado correctamente.")
        except Exception as e:
            logging.error(f"Error configurando el cliente de DeepSeek: {e}")
elif MODEL == "openai":
    logging.info("Usando modelo Azure OpenAI para la generación de contenido.")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        logging.error("Error: Las variables de entorno AZURE_OPENAI_API_KEY o AZURE_OPENAI_ENDPOINT no están configuradas.")
    else:
        try:
            client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2025-02-01-preview"
            )
        except Exception as e:
            logging.error(f"Error configurando el cliente de OpenAI: {e}")


# --- Constantes de Estado y TTL ---
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_NOT_FOUND = "not_found" # Estado implícito si no existe la key
GEMINI_ERROR_MARKER = "ERROR_PROCESSING_GEMINI" # Marcador en el resultado
DEEPSEEK_ERROR_MARKER = "ERROR_PROCESSING_DEEPSEEK" # Marcador en el resultado
OPENAI_ERROR_MARKER = "ERROR_PROCESSING_OPENAI" # Marcador en el resultado

# --- Modelos Pydantic ---
class TallyOption(BaseModel):
    id: str
    text: str
    
class TallyField(BaseModel):
    key: str
    label: Optional[str]
    value: Any
    type: str
    options: Optional[List[TallyOption]] = None

class TallyResponseData(BaseModel):
    responseId: str
    submissionId: str
    formName: str
    fields: List[TallyField]

class TallyWebhookPayload(BaseModel):
    eventId: str
    eventType: str
    data: TallyResponseData

# Placeholder model for PUT data (adjust as needed)
class UpdateResultPayload(BaseModel):
    new_result: str
    reason: Optional[str] = None

# Inicializar el cliente de OpenAI

# --- Endpoints FastAPI ---
@app.get("/items/{submission_id}")
async def fetch_item(item_id: str):

    # --- Configuración base de datos ---
    try:
        host = os.getenv("POSTGRES_HOST")
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        port = int(os.getenv("POSTGRES_PORT", 5432))  # Usa 5432 como valor por defecto si no se encuentra
    except KeyError as e:
        logging.error(f"Missing environment variable: {e}")
        return {"error": f"Missing environment variable: {e}"}, 500 

    try:
        # Conectar a la base de datos PostgreSQL con psycopg2   
        conn = psycopg2.connect(
                host=host,
                database=dbname,
                user=user,
                password=password,
                port=port,
                sslmode="require"
            )
        cur = conn.cursor()
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return {"error": f"Error connecting to the database: {e}"}, 500
    
    cur.execute("SELECT * FROM Items WHERE id = ?", (item_id,))
    row = cur.fetchone()
    if not row:
        logging.warning(f"Item with id {item_id} not found.")
        return {"error": "Item not found"}, 404
    return 

@app.get("/")
async def root():
    """Endpoint raíz simple para verificar que la app funciona."""
    return {"message": "Hola! Soy el procesador de Tally a Gemini."}