# 🚀 GUÍA DE ARRANQUE RÁPIDO

## Requisitos del Sistema

- **Python**: 3.10 o superior (recomendado 3.11+)
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **Hardware**:
  - CPU: Cualquier procesador moderno
  - GPU: NVIDIA con CUDA (opcional, mejora rendimiento)
  - RAM: 4GB mínimo, 8GB recomendado
  - Webcam o archivo de video

---

## Instalación Paso a Paso

### 1. Clonar/Descargar el Proyecto

```bash
git clone <url-del-repositorio> FallDetector
cd FallDetector
```

### 2. Crear Entorno Virtual (IMPORTANTE)

**Windows (PowerShell):**
```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
py -3 -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

O manualmente:
```bash
pip install ultralytics opencv-python numpy pyyaml
```

### 4. Verificar Instalación

```bash
python -c "import ultralytics; import cv2; print('✓ Instalación correcta')"
```

---

## Ejecutar el Programa

### Comando Básico (Webcam)

```bash
python main.py --source webcam --show
```

### Con Archivo de Video

```bash
python main.py --source video --path ruta/al/video.mp4 --show
```

### Todos los Parámetros

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--source` | `webcam` o `video` | `webcam` |
| `--path` | Ruta al video (si source=video) | - |
| `--camera` | Índice de cámara (0, 1, 2...) | `0` |
| `--model` | Modelo YOLO a usar | `yolo11n-pose.pt` |
| `--show` | Mostrar ventana de video | No |
| `--verbose` | Logs detallados en consola | No |
| `--output` | Carpeta para logs | `logs/` |
| `--no-log` | Desactivar logging a archivo | No |

### Ejemplos de Uso

```bash
# Webcam con visualización
python main.py --source webcam --show

# Segunda cámara
python main.py --source webcam --camera 1 --show

# Video con logs detallados
python main.py --source video --path test.mp4 --show --verbose

# Sin ventana (solo logs)
python main.py --source webcam --output resultados/
```

---

## Archivos Generados

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `YYYYMMDD_HHMMSS.csv` | `logs/` | Log CSV con todas las detecciones |
| `YYYYMMDD_HHMMSS.json` | `logs/` | Log JSON completo |
| `yolo11n-pose.pt` | Raíz del proyecto | Modelo (descargado automáticamente) |

### Formato del CSV

```csv
timestamp,risk_state,confirmed_state,risk_score,quality_score,torso_angle,reason,...
```

---

## Controles

- **`Q`**: Salir del programa
- **`ESC`**: Salir del programa (alternativo)

---

## Problemas Comunes y Soluciones

### ❌ "No module named 'numpy'" o similar

**Causa**: No estás usando el entorno virtual correcto.

**Solución**:
```bash
# Windows
.\venv\Scripts\python.exe main.py --source webcam --show

# O activa el venv primero
.\venv\Scripts\Activate.ps1
python main.py --source webcam --show
```

### ❌ "Could not open webcam: 0"

**Causas posibles**:
- Webcam en uso por otra aplicación (Zoom, Teams, etc.)
- Índice de cámara incorrecto
- Permisos de cámara denegados

**Soluciones**:
```bash
# Probar otra cámara
python main.py --source webcam --camera 1 --show

# Verificar cámaras disponibles
python -c "import cv2; [print(f'Cámara {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### ❌ "CUDA out of memory" o rendimiento lento

**Solución**: Forzar uso de CPU
```bash
# Editar main.py o usar modelo más pequeño
python main.py --source webcam --show --model yolo11n-pose.pt
```

### ❌ Modelo no se descarga

**Solución manual**:
```bash
pip install ultralytics --upgrade
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

### ❌ FPS muy bajos (< 5 FPS)

**Causas**: CPU lento, modelo grande, alta resolución

**Soluciones**:
- Usar modelo nano: `--model yolo11n-pose.pt`  
- Reducir resolución en `config/scheduler.yaml`
- Usar GPU si disponible

---

## Modelo Utilizado

### ✅ YOLO11 Pose (Confirmado)

**Archivo**: `core/inference_backend.py`  
**Línea 104**: 
```python
model_path: str = "yolo11n-pose.pt"
```

**Línea 125**:
```python
self.model = YOLO(self.model_path)
```

El modelo **YOLO11n-pose** es el modelo de pose estimation más reciente de Ultralytics, optimizado para velocidad (nano) con 17 keypoints del formato COCO.

---

## Estructura del Proyecto

```
FallDetector/
├── main.py                 # Punto de entrada
├── requirements.txt        # Dependencias
├── docs/
│   └── START.md           # ← ESTÁS AQUÍ
├── config/
│   ├── thresholds.yaml    # Umbrales de clasificación
│   └── scheduler.yaml     # Configuración de modos
├── core/                   # Lógica principal
├── utils/                  # Herramientas auxiliares
├── tests/                  # Tests unitarios
├── logs/                   # Logs generados (automático)
└── venv/                   # Entorno virtual (creado por ti)
```

---

## Siguiente Paso

Una vez funcionando:

1. **Calibrar umbrales**: Edita `config/thresholds.yaml`
2. **Analizar logs**: Revisa los CSV generados en `logs/`
3. **Probar posturas**: De pie → Sentado → Tumbado

¡El sistema debería detectar caídas automáticamente! 🎉
