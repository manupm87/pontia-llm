# OpenAI Translator

Aplicación conversacional con Streamlit para traducir texto entre idiomas usando la API de OpenAI y el modelo `gpt-4.1-nano`.

La app está diseñada como una pequeña aplicación de producto: separa interfaz, configuración, estado conversacional y cliente de OpenAI. No usa LangChain.

## Funcionalidad

- Traducción desde un idioma origen a un idioma destino mediante selectores.
- Detección automática del idioma de origen.
- Advertencia cuando el idioma seleccionado no coincide con el texto introducido.
- Selección de registro: neutral, formal, informal, técnico o marketing.
- Opción para preservar formato del texto original.
- Historial conversacional en la sesión de Streamlit.
- Exportación del historial a Markdown.
- Salida estructurada con JSON Schema para que la respuesta del modelo sea validable desde código.

## Estructura

```text
translator_openai/
├── app.py
├── core/
│   ├── config.py
│   ├── languages.py
│   ├── models.py
│   ├── openai_translator.py
│   ├── state.py
│   └── ui.py
├── .env.template
├── .streamlit/config.toml
├── Makefile
├── README.md
└── requirements.txt
```

## API key

Crea un archivo `.env` a partir de `.env.template`:

```bash
cp .env.template .env
```

Rellena tu clave:

```text
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4.1-nano"
OPENAI_TIMEOUT_SECONDS="45"
```

También puedes exportar `OPENAI_API_KEY` como variable de entorno si prefieres no usar `.env`.

## macOS y Linux

Desde esta carpeta:

```bash
cd sesion_02/translator_openai
make setup
make run
```

Por defecto, el `Makefile` usa `python3.11`. Si tu binario de Python se llama de otra manera:

```bash
make setup PYTHON=python3
make run
```

## Windows

Desde PowerShell, en esta carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.template .env
streamlit run app.py
```

Si usas `python` en lugar de `py`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Comandos útiles

```bash
make check
```

Verifica sintaxis e imports principales.

```bash
make clean
```

Elimina el entorno virtual local.

## Notas de implementación

La llamada a OpenAI está encapsulada en `core/openai_translator.py`. La app usa la API `Responses` con `text.format.type = "json_schema"` para pedir una respuesta estructurada con:

- `translation`
- `detected_source_language`
- `source_language`
- `target_language`
- `notes`

Esto evita depender de texto libre para alimentar la interfaz y facilita evolucionar la aplicación hacia validaciones, almacenamiento o integraciones posteriores.
