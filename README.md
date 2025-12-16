# 🚨 Fall Detection System - YOLO11 Pose

**Sistema de detección de caídas basado en YOLO11 Pose Estimation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11](https://img.shields.io/badge/YOLO-11--Pose-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TFG** - Sistema diseñado para detectar caídas y posturas de riesgo en entornos asistenciales, utilizando razonamiento geométrico sobre keypoints en lugar de clasificación end-to-end.

> [!WARNING]
> **Aviso Legal - Sistema Experimental**
> 
> Este proyecto es un trabajo de investigación académica (TFG).
> 
> **NO es un dispositivo médico** y no debe usarse como:
> - Sistema de diagnóstico clínico
> - Sustituto de supervisión humana
> - Sistema de seguridad crítico sin respaldo
> 
> El sistema puede fallar en detectar caídas o generar falsos positivos.
> Úselo bajo su responsabilidad y siempre con supervisión humana apropiada.

---

## 🚀 ¡ARRANQUE RÁPIDO!

```bash
# 1. Clonar e instalar
git clone <repo-url> FallDetector
cd FallDetector
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# 2. Ejecutar Web Dashboard (recomendado)
python web_server.py
# Abrir: http://localhost:8000
```

> **Cámara**: Usa Intel RealSense D435i automáticamente. Sin RealSense, usa webcam.

---

## 📋 Tabla de Contenidos

- [Motivación](#-motivación)
- [Características](#-características)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Arquitectura](#-arquitectura)
- [Calibración](#-calibración)
- [Limitaciones](#-limitaciones)

---

## 🎯 Motivación

### ¿Por qué Pose + Razonamiento?

Los detectores de caídas tradicionales suelen ser:
- Clasificadores binarios entrenados end-to-end
- Cajas negras difíciles de explicar
- Dependientes de datos de entrenamiento específicos

**Este sistema usa un enfoque diferente:**

1. **YOLO-Pose** extrae keypoints (pose estimation pre-entrenada)
2. **Razonamiento geométrico** analiza la postura (ángulos, alturas, proporciones)
3. **Depth 2.5D** mide altura real sobre el suelo (con RealSense)
4. **Confirmación temporal** evita falsos positivos

**Ventajas:**
- ✅ **Explicabilidad**: Cada decisión tiene una razón (`FLOOR_CONTACT + DROP_EVENT`)
- ✅ **Generalización**: No depende de dataset específico de caídas
- ✅ **Calibrable**: Umbrales en metros, ajustables sin reentrenar
- ✅ **Sofá/cama ≠ suelo**: Distingue superficies elevadas del suelo real

---

## ✨ Características

### Estados de Salida
| Estado | Significado | Color |
|--------|-------------|-------|
| `OK` | Normal - de pie o en sofá/cama | 🟢 Verde |
| `ANALYZING` | Evaluando o depth no confiable | 🟡 Naranja |
| `FALL` | En suelo confirmado (≥1 segundo) | 🔴 Rojo |

### Posturas Detectadas
- **LYING**: Persona tumbada (horizontal)
- **SITTING_FLOOR**: Sentado en el suelo
- **ALL_FOURS**: A cuatro patas
- **KNEELING**: Arrodillado
- **NORMAL**: De pie, caminando, sentado en silla/sofá

### Características Técnicas
- 📐 **Depth 2.5D**: Altura real sobre suelo con RANSAC floor plane
- 🛋️ **Discriminación sofá/cama**: No alerta si hip_height 35-80cm
- 📉 **Detección de caída**: vertical_drop > 35cm en 1.2s
- ⏱️ **Confirmación temporal**: 1 segundo de persistencia para confirmar
- 🎯 **Quality gates**: No confirma rojo si depth no es confiable

---

## 🚀 Instalación

### Requisitos
- Python 3.10+
- Intel RealSense D435i (recomendado) o webcam
- GPU opcional (también funciona en CPU)

### Pasos

```bash
# Clonar repositorio
git clone <repo-url> FallDetector
cd FallDetector

# Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Modelo YOLO-Pose se descarga automáticamente en primera ejecución
```

---

## 🎮 Uso

### Web Dashboard (Recomendado)

```bash
python web_server.py
```

Abrir en navegador: **http://localhost:8000**

Muestra:
- Video RGB con esqueleto superpuesto
- Video Depth colorizado
- Estado en tiempo real (OK/ANALYZING/FALL)
- Métricas: FPS, confianza, altura de cadera

Para cerrar: **Ctrl+C**

### Modo CLI (Alternativo)

```bash
# Webcam con visualización
python main.py --source webcam --show

# Archivo de video
python main.py --source video --path video.mp4 --show
```

### Opciones

| Parámetro | Descripción |
|-----------|-------------|
| `--no-realsense` | Forzar uso de webcam |
| `--camera N` | Índice de webcam (defecto: 0) |
| `--port N` | Puerto del servidor web (defecto: 8000) |

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                          MAIN.PY                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Frame Source │───▶│   Pose       │───▶│   Quality    │      │
│  │  (OpenCV)    │    │  Estimator   │    │  Assessor    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Inference    │    │   Feature    │───▶│  Classifier  │      │
│  │  Backend     │    │  Extractor   │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                   │               │
│                             ▼                   ▼               │
│                      ┌──────────────┐    ┌──────────────┐      │
│                      │   Temporal   │◀──▶│  Scheduler   │      │
│                      │   Analyzer   │    │  (Adaptive)  │      │
│                      └──────────────┘    └──────────────┘      │
│                             │                                   │
│              ┌──────────────┼──────────────┐                   │
│              ▼              ▼              ▼                   │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│       │  Output  │   │   Viz    │   │  Logger  │               │
│       │Publisher │   │          │   │ CSV/JSON │               │
│       └──────────┘   └──────────┘   └──────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Diseño para Escalabilidad

Las **abstracciones** permiten cambiar componentes sin reescribir:

| Componente | PC (Actual) | ROS2 (Futuro) | Jetson (Futuro) |
|------------|-------------|---------------|-----------------|
| Frame Source | `OpenCVFrameSource` | `ROS2ImageSource` | `DeepStreamSource` |
| Inference | `UltralyticsBackend` | `UltralyticsBackend` | `TensorRTBackend` |
| Output | `ConsolePublisher` | `ROS2Publisher` | `ROS2Publisher` |

---

## 🎚️ Calibración

### Archivos de Configuración

```
config/
├── thresholds.yaml    # Umbrales de clasificación y calidad
└── scheduler.yaml     # Configuración del scheduler adaptativo
```

### Umbrales Principales (`thresholds.yaml`)

```yaml
pose:
  # Ángulo del torso (grados desde horizontal)
  torso_angle_lying: 25.0     # < 25° = tumbado
  torso_angle_standing: 70.0  # > 70° = de pie/normal
  
  # Aspect ratio del bounding box
  aspect_ratio_lying: 1.8     # > 1.8 = orientación horizontal

quality:
  # Requisitos de calidad
  torso_missing_penalty: 0.2  # Sin torso visible → quality × 0.2
  min_quality_for_confirmation: 0.4  # Quality < 0.4 → no confirma NEEDS_HELP
```

### Scheduler (`scheduler.yaml`)

```yaml
modes:
  LOW_POWER:
    fps: 2              # Bajo consumo
    resolution_scale: 0.5
  CHECKING:
    fps: 12             # Verificando
  CONFIRMING:
    fps: 15             # Máxima atención

transitions:
  # REGLA CLAVE: UNKNOWN + riesgo elevado → CHECKING (nunca LOW_POWER)
  to_checking:
    unknown_with_risk: true
    unknown_risk_threshold: 0.4
```

### Cómo Calibrar

1. **Recoger logs**: Ejecutar con videos de prueba
2. **Analizar CSV**: Revisar `torso_angle`, `risk_score`, `reason`
3. **Ajustar umbrales**: Modificar YAML según observaciones
4. **No requerir recompilación**: Los cambios aplican al reiniciar

---

## 🤖 Migración a ROS2

El sistema está **preparado** para ROS2 con interfaces abstractas.

### Pasos de Migración

1. **Implementar `ROS2ImageSource`** en `core/frame_source.py`:
```python
class ROS2ImageSource(FrameSource):
    def __init__(self, topic: str = "/camera/image_raw"):
        self.subscription = node.create_subscription(
            Image, topic, self.callback, 10
        )
    
    def get_frame(self) -> Optional[FrameData]:
        # Convertir ROS Image a numpy
        return cv_bridge.imgmsg_to_cv2(self.latest_msg)
```

2. **Implementar `ROS2Publisher`** en `core/outputs.py`:
```python
class ROS2Publisher(OutputPublisher):
    def __init__(self):
        self.state_pub = node.create_publisher(FallState, '/fall_detector/state', 10)
    
    def publish(self, state: SystemState):
        msg = FallState()
        msg.state = state.confirmed_state.value
        msg.risk_score = state.risk_score
        self.state_pub.publish(msg)
```

3. **Crear nodo ROS2** que use el pipeline existente

### Estructura ROS2 Propuesta
```
fall_detector_ros/
├── fall_detector_ros/
│   ├── __init__.py
│   ├── detector_node.py
│   └── ros2_adapters.py
├── msg/
│   └── FallState.msg
├── launch/
│   └── detector.launch.py
└── package.xml
```

---

## 🔧 Migración a Jetson

Para despliegue en **NVIDIA Jetson** (Nano, Xavier, Orin):

### Paso 1: Exportar a TensorRT

```bash
# En Jetson (o con TensorRT instalado)
yolo export model=yolo11n-pose.pt format=engine device=0
```

### Paso 2: Implementar `TensorRTBackend`

```python
class TensorRTBackend(InferenceBackend):
    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
    
    def infer(self, frame: np.ndarray) -> List[PoseDetection]:
        # Preprocessing
        input_tensor = preprocess(frame)
        # TensorRT inference
        outputs = self.context.execute_v2(...)
        # Postprocessing
        return parse_outputs(outputs)
```

### Paso 3: Usar DeepStream (Opcional)

Para máximo rendimiento con múltiples cámaras:
- Usar DeepStream SDK para pipelines de video
- Hardware-accelerated decoding/encoding
- Mejor eficiencia energética

### Consideraciones Jetson

| Aspecto | Recomendación |
|---------|---------------|
| Modelo | `yolo11n-pose` (nano) para tiempo real |
| FP16 | Habilitar para 2x speedup |
| Batch | 1 para mínima latencia |
| Memoria | Reservar suficiente para TensorRT |

---

## ⚠️ Limitaciones

### Actuales

1. **Tracking de ID**:
   - Si hay múltiples personas, se selecciona una por frame
   - Podría confundir si cambian de posición

2. **Contexto espacial**:
   - No "sabe" dónde están los muebles
   - Distingue sofá/cama del suelo por altura (depth), no por semántica

3. **Oclusión**:
   - Si el torso no es visible, quality baja
   - Puede no detectar postura correctamente

4. **Iluminación variable**:
   - Depth puede fallar con luz muy baja
   - En esos casos el sistema pasa a ANALYZING (naranja)

### Mejoras Futuras

- [ ] Añadir tracking con IDs persistentes
- [ ] Mapa semántico del entorno
- [ ] Detector de actividad (caída vs transición vs acostado)
- [ ] Fusión con sensores adicionales (audio, PIR)

---

## 📊 Estructura del Proyecto

```
FallDetector/
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias Python
├── README.md               # Esta documentación
│
├── docs/
│   └── START.md           # 📖 Tutorial de arranque rápido
│
├── config/
│   ├── thresholds.yaml     # Umbrales de clasificación
│   └── scheduler.yaml      # Configuración del scheduler
│
├── core/
│   ├── frame_source.py     # Abstracción de fuente de frames
│   ├── inference_backend.py # ✅ YOLO11 Pose (línea 104, 125)
│   ├── pose_estimator.py   # Wrapper YOLO + selección persona
│   ├── quality.py          # Evaluación de calidad
│   ├── features.py         # Extracción de features
│   ├── classifier.py       # Clasificación de poses
│   ├── temporal.py         # Confirmación temporal adaptativa
│   ├── scheduler.py        # Scheduler adaptativo
│   └── outputs.py          # Publicación de resultados
│
├── utils/
│   ├── geometry.py         # Funciones geométricas
│   ├── dashboard.py        # 🎨 UI Dashboard (esqueleto coloreado)
│   ├── viz.py              # Visualización básica
│   └── logging.py          # Logging explicable
│
├── tests/
│   ├── test_geometry.py
│   ├── test_quality.py
│   └── test_temporal.py
│
└── scripts/
    ├── run_webcam.bat
    └── run_video.bat
```

---

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) por YOLO
- COCO Dataset por el formato de keypoints

---

**Desarrollado como Trabajo de Fin de Grado**
