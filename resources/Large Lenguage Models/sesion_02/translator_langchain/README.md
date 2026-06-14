# LangChain Translator

Aplicación conversacional con Streamlit para traducir texto entre idiomas usando LangChain como capa de abstracción sobre distintos proveedores.

El usuario puede elegir proveedor y modelo desde la barra lateral, manteniendo la misma experiencia de traducción para todos los backends soportados.

## Funcionalidad

- Selector de proveedor: OpenAI o Google Gemini.
- Selector de modelo según proveedor.
- Traducción desde un idioma origen a un idioma destino.
- Detección automática del idioma de origen.
- Advertencia cuando el idioma seleccionado no coincide con el texto introducido.
- Streaming de la traducción vía LangChain.
- Selección de registro: neutral, formal, informal, técnico o marketing.
- Historial conversacional, exportación a Markdown y paginación visual.

## API keys

```bash
cp .env.template .env
```

Rellena las claves de los proveedores que quieras usar:

```text
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="..."
LANGCHAIN_PROVIDER="OpenAI"
LANGCHAIN_MODEL="gpt-4.1-nano"
LANGCHAIN_TIMEOUT_SECONDS="45"
```

## macOS y Linux

```bash
cd sesion_02/translator_langchain
make setup
make run
```

## Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.template .env
streamlit run app.py
```

## Estructura

La lógica específica de LangChain está en `core/langchain_translator.py`. Ahí se inicializa el wrapper correspondiente con `init_chat_model`, se hace detección estructurada del idioma y se genera la traducción en streaming.
