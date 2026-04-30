# SIGPAC Sentinel Backend

FastAPI backend para procesado de imágenes Sentinel-2 e integración SIGPAC.

## 🚀 Deploy en Render

1. Sube este directorio a un repositorio GitHub
2. Ve a [render.com](https://render.com) → New → Web Service
3. Conecta tu repo
4. Render detecta automáticamente el `render.yaml`
5. Añade las variables de entorno:
   - `COPERNICUS_USER` → tu email de [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
   - `COPERNICUS_PASS` → tu contraseña

## 📦 Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio |
| GET | `/sigpac/parcela` | Geometría parcela SIGPAC |
| GET | `/sentinel/buscar` | Buscar imágenes disponibles |
| GET | `/indices/lista` | Índices disponibles |
| GET | `/indice/calcular` | Calcular y descargar índice |
| GET | `/cache/info` | Info del caché |
| DELETE | `/cache/limpiar` | Limpiar caché antiguo |

## 🌱 Índices disponibles

- **NDVI** - Vegetación (rojo/infrarrojo)
- **NDWI** - Agua (verde/infrarrojo)
- **EVI** - Vegetación avanzada
- **NDRE** - Red Edge (viñedos, frutales)
- **SAVI** - Suelo ajustado

## 🔑 Credenciales gratuitas Copernicus

1. Regístrate en https://dataspace.copernicus.eu
2. Cuenta 100% gratuita
3. Acceso completo a Sentinel-2

## ⚙️ Local

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Docs en: http://localhost:8000/docs
```

## 📝 Notas

- Sin credenciales Copernicus, el backend funciona en **modo DEMO**
- El caché evita redescargas: imágenes se guardan en `/cache`
- SIGPAC WFS es público, no necesita credenciales
