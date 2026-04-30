from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIGPAC Sentinel API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── APIs oficiales actualizadas ──────────────────────────────────────────────
# Nueva API oficial FEGA (sigpac-hubcloud.es)
SIGPAC_CONSULTA_URL   = "https://sigpac-hubcloud.es/servicioconsultassigpac/query"
SIGPAC_OGC_URL        = "https://sigpac-hubcloud.es/ogcapi/collections/recintos/items"

# Copernicus
COPERNICUS_TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")

_token_cache = {"token": None, "expires_at": 0}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def get_copernicus_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    if not COPERNICUS_USER or not COPERNICUS_PASS:
        raise HTTPException(status_code=500, detail="Credenciales Copernicus no configuradas.")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COPERNICUS_TOKEN_URL,
            data={"grant_type": "password", "username": COPERNICUS_USER,
                  "password": COPERNICUS_PASS, "client_id": "cdse-public"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        return _token_cache["token"]


def cache_key(prefix: str, **kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()


def decode_band(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr / 10000.0


def calcular_formula(nombre: str, b: dict) -> np.ndarray:
    eps = 1e-10
    if nombre == "NDVI":
        return (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    elif nombre == "NDWI":
        return (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps)
    elif nombre == "EVI":
        return 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1 + eps)
    elif nombre == "NDRE":
        return (b["B08"] - b["B05"]) / (b["B08"] + b["B05"] + eps)
    elif nombre == "SAVI":
        return 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5 + eps)
    raise ValueError(f"Índice desconocido: {nombre}")


INDICES = {
    "NDVI": {"descripcion": "Normalized Difference Vegetation Index", "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "bandas": ["B04", "B08"]},
    "NDWI": {"descripcion": "Normalized Difference Water Index",      "cmap": "Blues",  "vmin": -1, "vmax": 1, "bandas": ["B03", "B08"]},
    "EVI":  {"descripcion": "Enhanced Vegetation Index",              "cmap": "YlGn",   "vmin": -1, "vmax": 1, "bandas": ["B02", "B04", "B08"]},
    "NDRE": {"descripcion": "Normalized Difference Red Edge",         "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "bandas": ["B05", "B08"]},
    "SAVI": {"descripcion": "Soil-Adjusted Vegetation Index",         "cmap": "YlGn",   "vmin": -1, "vmax": 1, "bandas": ["B04", "B08"]},
}


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "copernicus_configured": bool(COPERNICUS_USER and COPERNICUS_PASS),
    }


@app.get("/sigpac/parcela")
async def get_parcela(
    provincia: int = Query(..., description="Código provincia (ej: 41 = Sevilla)"),
    municipio: int = Query(..., description="Código municipio (ej: 74)"),
    agregado: int = Query(0, description="Agregado (normalmente 0)"),
    zona: int = Query(0, description="Zona (normalmente 0)"),
    poligono: int = Query(..., description="Número polígono"),
    parcela: int = Query(..., description="Número parcela"),
):
    """
    Obtiene la geometría de una parcela SIGPAC.
    Referencia completa: provincia/municipio/agregado/zona/poligono/parcela
    """
    ck = cache_key("sigpac3", prov=provincia, mun=municipio, agr=agregado, zon=zona, pol=poligono, par=parcela)
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"

    if cache_file.exists():
        logger.info(f"Cache hit SIGPAC: {ck}")
        return JSONResponse(content=json.loads(cache_file.read_text()))

    # Referencia SIGPAC: provincia(2) municipio(3) agregado(1) zona(1) poligono(5) parcela(5) recinto(4)
    ref = f"{provincia:02d}{municipio:03d}{agregado:01d}{zona:01d}{poligono:05d}{parcela:05d}0001"
    logger.info(f"Referencia SIGPAC: {ref}")

    try:
        async with httpx.AsyncClient(
            timeout=30,
            headers={"Accept": "application/geo+json, application/json"},
            follow_redirects=True,
        ) as client:

            # ── Método 1: Servicio de Consultas por referencia completa ──────
            url_consulta = f"{SIGPAC_CONSULTA_URL}/recinfo/{ref}.geojson"
            logger.info(f"Método 1 - URL: {url_consulta}")
            resp = await client.get(url_consulta)
            logger.info(f"Método 1 - Status: {resp.status_code}")

            if resp.status_code == 200:
                raw = resp.json()
                if raw.get("type") == "Feature":
                    data = {"type": "FeatureCollection", "features": [raw]}
                elif raw.get("type") == "FeatureCollection":
                    data = raw
                else:
                    data = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": raw.get("geometry"), "properties": raw}]}
                if data.get("features") and data["features"][0].get("geometry"):
                    cache_file.write_text(json.dumps(data))
                    return JSONResponse(content=data)

            # ── Método 2: OGC API Features con filtro completo ───────────────
            params_ogc = {
                "f": "application/geo+json",
                "limit": "5",
                "filter": (
                    f"provincia={provincia} AND municipio={municipio} AND "
                    f"agregado={agregado} AND zona={zona} AND "
                    f"poligono={poligono} AND parcela={parcela}"
                ),
                "filter-lang": "cql-text",
            }
            resp2 = await client.get(SIGPAC_OGC_URL, params=params_ogc)
            logger.info(f"Método 2 OGC - Status: {resp2.status_code}")

            if resp2.status_code == 200:
                data = resp2.json()
                if data.get("features") and data["features"][0].get("geometry"):
                    cache_file.write_text(json.dumps(data))
                    return JSONResponse(content=data)

        raise HTTPException(
            status_code=404,
            detail=(
                f"Parcela no encontrada: {provincia}/{municipio}/{agregado}/{zona}/{poligono}/{parcela}. "
                f"Verifica que los códigos son correctos."
            )
        )

    except HTTPException:
        raise
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error conectando con SIGPAC: {str(e)}")


@app.get("/sigpac/punto")
async def get_parcela_por_punto(
    lat: float = Query(..., description="Latitud WGS84 (ej: 37.38)"),
    lon: float = Query(..., description="Longitud WGS84 (ej: -5.97)"),
):
    """Obtiene la parcela SIGPAC que contiene un punto geográfico. Más fiable que buscar por códigos."""
    ck = cache_key("sigpac_punto", lat=round(lat, 6), lon=round(lon, 6))
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"

    if cache_file.exists():
        return JSONResponse(content=json.loads(cache_file.read_text()))

    url = f"{SIGPAC_CONSULTA_URL}/recinfobypoint/4326/{lon}/{lat}.geojson"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("features"):
            raise HTTPException(status_code=404, detail="No se encontró parcela en esas coordenadas.")

        cache_file.write_text(json.dumps(data))
        return JSONResponse(content=data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error SIGPAC: {str(e)}")


@app.get("/sentinel/buscar")
async def buscar_imagenes(
    bbox: str = Query(...),
    fecha_inicio: str = Query(...),
    fecha_fin: str = Query(...),
    max_nubosidad: float = Query(30.0),
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox inválido")

    footprint = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and ContentDate/Start gt {fecha_inicio}T00:00:00.000Z "
            f"and ContentDate/Start lt {fecha_fin}T23:59:59.000Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_nubosidad}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
        "$expand": "Attributes",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(COPERNICUS_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        productos = []
        for item in data.get("value", []):
            cloud = next((a["Value"] for a in item.get("Attributes", []) if a["Name"] == "cloudCover"), None)
            productos.append({
                "id": item["Id"], "nombre": item["Name"],
                "fecha": item["ContentDate"]["Start"][:10],
                "nubosidad": round(cloud, 1) if cloud is not None else None,
                "size_mb": round(item.get("ContentLength", 0) / 1e6, 1),
            })
        return {"total": len(productos), "productos": productos}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error Copernicus: {e}")


@app.get("/indices/lista")
async def lista_indices():
    return {k: {"descripcion": v["descripcion"], "bandas": v["bandas"]} for k, v in INDICES.items()}


@app.get("/indice/calcular")
async def calcular_indice(
    producto_id: str = Query(...),
    indice: str = Query(...),
    bbox: Optional[str] = Query(None),
    formato: str = Query("png"),
):
    indice = indice.upper()
    if indice not in INDICES:
        raise HTTPException(status_code=400, detail=f"Índice desconocido. Disponibles: {list(INDICES.keys())}")

    cfg = INDICES[indice]
    ck = cache_key("indice", pid=producto_id, idx=indice, bbox=bbox or "")
    cache_png = CACHE_DIR / f"{ck}.png"
    cache_stats = CACHE_DIR / f"{ck}_stats.json"

    if cache_png.exists() and formato == "png":
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")
    if cache_stats.exists() and formato == "stats":
        return JSONResponse(content=json.loads(cache_stats.read_text()))

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo(indice, cfg, cache_png, cache_stats, formato)

    base = "https://download.dataspace.copernicus.eu/odata/v1"
    arrays = {}

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
        follow_redirects=True,
    ) as client:
        for banda in cfg["bandas"]:
            urls = [
                f"{base}/Products({producto_id})/Nodes({producto_id}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({banda}.jp2)/$value",
                f"{base}/Products({producto_id})/Nodes({producto_id}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes(R20m)/Nodes({banda}.jp2)/$value",
            ]
            ok = False
            for url in urls:
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        arrays[banda] = decode_band(r.content)
                        ok = True
                        break
                except Exception:
                    pass
            if not ok:
                return _demo(indice, cfg, cache_png, cache_stats, formato)

    shapes = [a.shape for a in arrays.values()]
    if len(set(shapes)) > 1:
        target = min(shapes, key=lambda s: s[0] * s[1])
        arrays = {
            k: np.array(Image.fromarray(v).resize((target[1], target[0]), Image.BILINEAR), dtype=np.float32)
            if v.shape != target else v for k, v in arrays.items()
        }

    resultado = np.clip(calcular_formula(indice, arrays), cfg["vmin"], cfg["vmax"])
    return _render(resultado, indice, cfg, cache_png, cache_stats, formato, demo=False)


def _demo(indice, cfg, cache_png, cache_stats, formato):
    np.random.seed(42)
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    base = 0.3 + 0.4 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.15)
    resultado = np.clip(base + np.random.normal(0, 0.05, (256, 256)), cfg["vmin"], cfg["vmax"]).astype(np.float32)
    return _render(resultado, indice, cfg, cache_png, cache_stats, formato, demo=True)


def _render(resultado, indice, cfg, cache_png, cache_stats, formato, demo=False):
    stats = {
        "indice": indice,
        "min": float(np.nanmin(resultado)),
        "max": float(np.nanmax(resultado)),
        "mean": float(np.nanmean(resultado)),
        "std": float(np.nanstd(resultado)),
    }
    if demo:
        stats["modo"] = "DEMO"
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    titulo = f"{indice} · {cfg['descripcion']}" + (" (DEMO)" if demo else "")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('#0a0f0d')
    ax.set_facecolor('#0a1a0d')
    im = ax.imshow(resultado, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#6b8f72')
    cbar.outline.set_edgecolor('#2a3d2e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#6b8f72', fontsize=9)
    ax.set_title(titulo, color='#e2ffe8', fontsize=12, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0f0d')
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/cache/info")
async def cache_info():
    files = list(CACHE_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return {"archivos": len(files), "total_mb": round(total_mb, 2)}


@app.delete("/cache/limpiar")
async def limpiar_cache(dias: int = Query(7)):
    cutoff = time.time() - dias * 86400
    eliminados = sum(1 for f in CACHE_DIR.glob("*") if f.is_file() and f.stat().st_mtime < cutoff and not f.unlink())
    return {"eliminados": eliminados}
