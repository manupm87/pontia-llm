"""Núcleo del asistente turístico de Tenerife.

Este paquete reúne la lógica reutilizable que consumen tanto el notebook
principal como la aplicación Streamlit:

- ``config``: configuración y parámetros del modelo.
- ``rag``: indexado del PDF y recuperación con citas (RAG).
- ``images``: extracción de las fotos del PDF e indexado por página.
- ``weather``: llamada a función externa ``get_weather`` (Open-Meteo).
- ``tools``: las herramientas que el LLM puede invocar.
- ``assistant``: bucle conversacional multiturno con tool calling.
"""
