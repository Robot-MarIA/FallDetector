# Tutorial de Arranque - Web Dashboard

## Requisitos

- Python 3.10+
- Intel RealSense D435i (o webcam como fallback)

## Instalacion

```bash
# Clonar repo y entrar
git clone <url-del-repo> FallDetector
cd FallDetector

# Crear y activar entorno virtual
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install pyrealsense2  # Para RealSense
```

## Ejecutar con RealSense (por defecto)

```bash
python web_server.py
```

Esto usa la RealSense con:
- **RGB**: 1280x720 @30fps
- **Depth**: 848x480 @30fps (alineado con RGB)

Si no hay RealSense, usa la webcam automáticamente.

## Abrir Dashboard

**http://localhost:8000**

## Cerrar el Servidor

Presiona **Ctrl+C** en la terminal.

## Caracteristicas 2.5D (Depth)

El sistema usa profundidad para:
- **Estimar el plano del suelo** (RANSAC)
- **Medir altura de cadera** sobre el suelo en metros
- **Distinguir sofá/cama vs suelo**
- **Detectar eventos de caída** (drop > 35cm)

### Estados

| Estado | Condicion |
|--------|-----------|
| OK | De pie o en superficie elevada (sofa/cama) |
| ANALYZING | Incertidumbre o depth no confiable |
| FALL | En suelo confirmado >= 1 segundo |

### Thresholds

Ver `config/thresholds.yaml` seccion `depth:` para ajustar.
