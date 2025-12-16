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

**→ [Leer Tutorial Completo de Arranque](docs/START.md) ←**

```bash
# Instalación rápida
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install ultralytics opencv-python pyyaml

# Ejecutar
python main.py --source webcam --show
```

---

## 📋 Tabla de Contenidos

- [Motivación](#-motivación)
- [Características](#-características)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Arquitectura](#-arquitectura)
- [Calibración](#-calibración)
- [Migración a ROS2](#-migración-a-ros2)
- [Migración a Jetson](#-migración-a-jetson)
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
3. **Confirmación temporal** evita falsos positivos

**Ventajas:**
- ✅ **Explicabilidad**: Cada decisión tiene una razón (`TORSO_HORIZONTAL + LOW_HEIGHT`)
- ✅ **Generalización**: No depende de dataset específico de caídas
- ✅ **Calibrable**: Umbrales ajustables sin reentrenar
- ✅ **Trazabilidad**: Logs detallados para análisis académico

---

## ✨ Características

### Estados de Salida
| Estado | Significado | Color |
|--------|-------------|-------|
| `OK` | Sin riesgo detectado | 🟢 Verde |
| `RISK` | Posible riesgo, requiere atención | 🟡 Naranja |
| `NEEDS_HELP` | Postura de riesgo confirmada | 🔴 Rojo |
| `UNKNOWN` | Información insuficiente | ⚪ Gris |

### Posturas Detectadas
- **LYING**: Persona tumbada (horizontal)
- **SITTING_FLOOR**: Sentado en el suelo
- **ALL_FOURS**: A cuatro patas
- **KNEELING**: Arrodillado
- **NORMAL**: De pie, caminando, sentado en silla

### Características Técnicas
- 🔄 **Temporalidad adaptativa**: Ventana de confirmación dinámica (1-5s)
- 📊 **Quality score**: Penalización severa sin torso visible
- ⚡ **Scheduler adaptativo**: 3 modos (LOW_POWER, CHECKING, CONFIRMING)
- 📝 **Logs explicables**: CSV/JSON con reason strings
- 🎯 **Selección de persona**: Bbox más grande o más centrado

---

## 🚀 Instalación

### Requisitos
- Python 3.8+
- Webcam o archivos de video
- GPU recomendada (también funciona en CPU)

### Pasos

```bash
# Clonar repositorio
git clone <repo-url>
cd FallDetector

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo YOLO-Pose (automático en primera ejecución)
# El modelo yolo11n-pose.pt se descarga automáticamente
```

### Verificar instalación
```bash
# Ejecutar tests
pytest tests/ -v
```

---

## 🎮 Uso Rápido

### Con Webcam
```bash
python main.py --source webcam --show
```

### Con Video
```bash
python main.py --source video --path ruta/al/video.mp4 --show
```

### Scripts de Acceso Rápido (Windows)
```bash
# Webcam
scripts\run_webcam.bat

# Video
scripts\run_video.bat C:\Videos\test.mp4
```

### Opciones Completas
```bash
python main.py --help

# Ejemplos:
python main.py --source webcam --show --verbose
python main.py --source video --path video.mp4 --output logs/
python main.py --source webcam --model yolo11s-pose.pt  # Modelo más preciso
```

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
