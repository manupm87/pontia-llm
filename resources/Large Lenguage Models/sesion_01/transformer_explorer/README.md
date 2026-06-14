# Transformer Explorer

Aplicación interactiva para explorar visualmente la arquitectura Transformer construida en `Transformer.ipynb`.

La app está pensada como apoyo docente: muestra el flujo encoder-decoder, permite seleccionar capas, hace zoom sobre componentes y explica sus subcapas. No entrena modelos, no ejecuta TensorFlow y no carga el modelo `translator/` en tiempo real.

## Requisitos

Necesitas:

- Node.js `^20.19.0` o `>=22.12.0`.
- npm, que normalmente viene incluido con Node.js.

## Instalar Node.js

### Windows

1. Descarga la versión LTS desde:

   ```text
   https://nodejs.org/
   ```

2. Ejecuta el instalador y deja marcada la opción para añadir Node.js al `PATH`.
3. Cierra y vuelve a abrir PowerShell.
4. Comprueba la instalación:

   ```powershell
   node --version
   npm --version
   ```

### macOS

Opción recomendada con instalador:

```text
https://nodejs.org/
```

También puedes usar Homebrew:

```bash
brew install node
node --version
npm --version
```

### Linux

Usa el método recomendado para tu distribución o instala Node.js desde:

```text
https://nodejs.org/
```

Comprueba la instalación:

```bash
node --version
npm --version
```

## Ejecutar la app

Desde la carpeta que contiene este README:

```bash
npm install
npm run dev
```

Abre en el navegador:

```text
http://localhost:5173/
```

Si estás en Windows con PowerShell:

```powershell
npm install
npm run dev
```

## Compilar la app

Para verificar que todo compila correctamente:

```bash
npm run build
```

Para previsualizar la versión compilada:

```bash
npm run preview
```

## Notas

- La carpeta `.node/`, si existe, es solo un runtime local usado durante desarrollo y no hace falta para ejecutar la app.
- La carpeta `node_modules/` se genera con `npm install` y no debe compartirse manualmente.
- La app usa datos didácticos precargados para explicar la traducción portugués-inglés del notebook.
