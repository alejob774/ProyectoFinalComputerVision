# 🏀 Proyecto: Análisis Automático de la Trayectoria de un Tiro de Baloncesto

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema automatizado que utiliza **visión por computador** y **técnicas de análisis geométrico** para detectar, analizar y visualizar la trayectoria de un tiro de baloncesto. Combina modelos de detección de objetos basados en **YOLOv8** con principios de cinemática para extraer métricas clave del movimiento del balón.

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema automatizado que detecte, analice y visualice la trayectoria de un tiro de baloncesto utilizando visión por computador y técnicas de análisis geométrico.

### Objetivos Específicos
- ✅ Detectar balón y aro usando YOLOv8 entrenado
- ✅ Clasificar el tipo de tiro mediante análisis de trayectoria
- ✅ Calcular métricas clave (punto más alto, ángulo, velocidad)
- ✅ Generar visualización profesional de la trayectoria
- ✅ Exportar resultados a Excel y imágenes integradas

## 🛠️ Arquitectura del Sistema

### 📁 Estructura de Archivos

```
proyecto-baloncesto/
│
├── 📊 predict_video_to_excel.py    # Procesamiento principal del video
├── 🎨 visualizer.py                # Generación de visualizaciones
├── 🔗 conjunto.py                  # Integración final de resultados
├── 🚀 main.py                      # Interfaz unificada de ejecución
├── ⚙️ best.pt                      # Modelo YOLOv8 entrenado
└── 📁 datasets/                    # Datasets de entrenamiento
```

## 🔄 Flujo de Trabajo Paso a Paso

### **Paso 1: Entrenamiento del Modelo YOLOv8**

**📝 Qué se hizo:**
- Se utilizaron dos datasets especializados de Roboflow para balón y aro
- Entrenamiento con arquitectura YOLOv8n durante 500 épocas
- Configuración: tamaño de imagen 640x640, paciencia 20, early stopping

**🎯 Para qué sirve:**
- Crear un modelo capaz de detectar balón y aro en videos de baloncesto
- Proporcionar la base para todo el análisis posterior

### **Paso 2: Procesamiento del Video** (`predict_video_to_excel.py`)

**📝 Qué se hace:**
```python
# Flujo del script:
1. Carga el modelo YOLOv8 entrenado (best.pt)
2. Procesa el video frame por frame
3. Para cada frame:
   - Detecta balón y aro usando YOLO
   - Almacena coordenadas (X,Y) del balón
   - Identifica el aro para calibración
4. Calcula métricas:
   - Punto más alto de la trayectoria
   - Última detección válida del balón
   - Ángulo del tiro: θ = arctan2((Y_last - Y_high), (X_high - X_last))
   - Velocidad horizontal (conversión píxeles→metros)
5. Exporta resultados a Excel
```

**🎯 Para qué sirve:**
- Extraer datos cuantitativos del movimiento del balón
- Proporcionar las métricas fundamentales para el análisis
- Crear base de datos estructurada en Excel

**📤 Output:** `resultado_tiro.xlsx`

### **Paso 3: Visualización de la Trayectoria** (`visualizer.py`)

**📝 Qué se hace:**
```python
# Proceso de visualización:
1. Carga el último frame donde aparece el balón
2. Dibuja sobre la imagen:
   - Línea VERDE: trayectoria completa del balón
   - Punto ROJO: última detección del balón
   - Punto AZUL: punto más alto de la trayectoria
3. Genera tabla resumen con métricas calculadas
4. Combina imagen + tabla en una visualización unificada
```

**🎯 Para qué sirve:**
- Proporcionar representación visual intuitiva del tiro
- Facilitar el análisis técnico para entrenadores y jugadores
- Comunicar resultados de manera clara y profesional

**📤 Output:** `visualizacion_con_tabla.png`

### **Paso 4: Integración Completa** (`conjunto.py`)

**📝 Qué se hace:**
- Combina todos los elementos anteriores en una salida integrada
- Asegura consistencia entre datos y visualización
- Produce la imagen final del análisis completo

**🎯 Para qué sirve:**
- Crear el producto final del análisis
- Proporcionar una herramienta completa para presentaciones
- Integrar todos los componentes del sistema

**📤 Output:** `salida_conjunto.png`

### **Paso 5: Ejecución Unificada** (`main.py`)

**📝 Qué se hace:**
- Proporciona una interfaz de línea de comandos unificada
- Permite ejecutar todo el flujo con un solo comando
- Gestiona la secuencia: procesamiento → visualización → integración

**🎯 Para qué sirve:**
- Simplificar el uso del sistema
- Automatizar el flujo completo de trabajo
- Facilitar la reproducibilidad

## 🔬 Fundamentos Teóricos Implementados

### 📐 Cinemática del Tiro
```python
# Movimiento parabólico (simplificado)
x(t) = v0 * cos(θ) * t
y(t) = v0 * sin(θ) * t - (1/2) * g * t²

# Cálculo práctico del ángulo
θ = arctan2((Y_last - Y_high), (X_high - X_last))
```

### 📏 Conversión de Unidades
```python
# Calibración usando el aro (diámetro real: 0.4572 m)
PIXELS_PER_METER = rim_width_pixels / 0.4572

# Cálculo de velocidad
Δx_metros = Δx_pix / PIXELS_PER_METER
Velocidad = Δx_metros / Δt
```

## 📊 Métricas Calculadas

| Métrica | Descripción | Fórmula/Base |
|---------|-------------|--------------|
| **Punto más alto** | Máxima altura en píxeles | `max(y_coordinates)` |
| **Última detección** | Posición final del balón | Último frame con detección |
| **Ángulo del tiro** | Inclinación de la trayectoria | `arctan2(Δy, Δx)` |
| **Velocidad horizontal** | Velocidad en eje X | `Δx_metros / Δt` |

## 🎯 Resultados Esperados

### ✅ Salidas del Sistema
1. **`resultado_tiro.xlsx`** - Datos tabulados de las métricas
2. **`visualizacion_con_tabla.png`** - Imagen con trayectoria y tabla
3. **`salida_conjunto.png`** - Resultado final integrado

### 📈 Ejemplo de Resultados
```
- Punto más alto: frame 34, (X = 445.78, Y = 52.27)
- Última detección: frame 45, (X = 229.86, Y = 134.04)
- Ángulo del tiro: 20.74°
- Velocidad horizontal: 1.923 m/s
```

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Ejecución Completa
```bash
python main.py --video video_tiro.mp4
```

### Opción 2: Ejecución por Módulos
```bash
# 1. Procesar video y extraer métricas
python predict_video_to_excel.py --video video_tiro.mp4

# 2. Generar visualización
python visualizer.py

# 3. Integrar resultados finales
python conjunto.py
```

## 💡 Aplicaciones y Beneficios

### 🏆 Para Entrenadores
- Análisis objetivo de la técnica de tiro
- Identificación de patrones de movimiento
- Base para correcciones técnicas

### 📊 Para Analistas Deportivos
- Datos cuantitativos para estudios
- Comparación entre diferentes jugadores
- Seguimiento de evolución temporal

### 🎓 Para Investigación
- Herramienta para estudios biomecánicos
- Base para desarrollo de tecnologías deportivas
- Plataforma para análisis avanzados 3D

## 🔮 Futuras Mejoras

1. **Análisis 3D** - Incorporar profundidad a las trayectorias
2. **Tiempo Real** - Procesamiento en vivo durante partidos
3. **Métricas Avanzadas** - Efecto Magnus, resistencia del aire
4. **Multi-tiro** - Análisis comparativo de múltiples lanzamientos

---

## 📞 Soporte y Contribuciones

Este proyecto representa una integración exitosa de visión por computador con análisis deportivo, demostrando cómo la inteligencia artificial puede proporcionar insights valiosos en el ámbito del baloncesto profesional y de formación.
