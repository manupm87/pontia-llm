# Gemini Translator

Aplicación conversacional con Streamlit para traducir texto entre idiomas usando la API de Google Gemini y el modelo `gemini-2.5-flash-lite`.

## Funcionalidad

- Traducción desde un idioma origen a un idioma destino.
- Detección automática del idioma de origen.
- Advertencia cuando el idioma seleccionado no coincide con el texto introducido.
- Streaming de la traducción.
- Selección de registro: neutral, formal, informal, técnico o marketing.
- Historial conversacional, exportación a Markdown y paginación visual.

## API key

```bash
cp .env.template .env
```

Rellena:

```text
GOOGLE_API_KEY="..."
GOOGLE_MODEL="gemini-2.5-flash-lite"
GOOGLE_TIMEOUT_SECONDS="45"
```

## macOS y Linux

```bash
cd sesion_02/translator_gemini
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

La lógica específica de Gemini está en `core/gemini_translator.py`. El resto de módulos separan configuración, estado, modelos de datos, catálogo de idiomas e interfaz.
