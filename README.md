# 📐 Calculadora de Longitud de Cuerda

## 🎯 Descripción del Proyecto

Este programa calcula la longitud real de una cuerda a partir de una fotografía usando métodos matemáticos avanzados como el **Método de Trapecio Compuesto** e **Interpolación Polinómica**. Es ideal para proyectos de física, matemáticas o ingeniería donde necesites medir objetos curvos con precisión.

## 🚀 Características Principales

- **Detección automática** de la cuerda en la imagen usando filtros Canny
- **Modo manual** para casos donde la detección automática falla
- **Interpolación con splines cúbicos** para suavizar la curva
- **Método de trapecio compuesto** con 500 subdivisiones para máxima precisión
- **Generación de funciones polinómicas** por segmentos
- **Visualizaciones completas** con 12 gráficos diferentes
- **Exportación a CSV** de todos los datos calculados
- **Interfaz amigable** con menús y emojis

## 📋 Requisitos del Sistema

### Librerías Necesarias
```bash
pip install opencv-python matplotlib scipy pandas numpy pillow
```

### Especificaciones de la Imagen
- **Formatos soportados**: JPG, PNG, BMP, TIFF
- **Resolución mínima**: 800x600 píxeles
- **Calidad**: Buena iluminación, fondo contrastante
- **Objeto**: Cuerda completamente visible y estirada

## 🛠️ Instalación

1. **Clona o descarga** el archivo del programa
2. **Instala las dependencias**:
   ```bash
   pip install opencv-python matplotlib scipy pandas numpy pillow
   ```
3. **Coloca tu imagen** en la misma carpeta que el script
4. **Ejecuta el programa**:
   ```bash
   python calculadora_cuerda.py
   ```

## 📸 Uso del Programa

### Paso 1: Preparar la Imagen
- Toma una foto clara de tu cuerda
- Asegúrate de que esté completamente visible
- Usa un fondo que contraste con la cuerda
- Guarda la imagen en la misma carpeta del programa

### Paso 2: Ejecutar el Análisis
1. Ejecuta el programa
2. Selecciona opción "1" del menú
3. El programa buscará automáticamente tu imagen
4. Si no la encuentra, te mostrará las imágenes disponibles

### Paso 3: Configurar Parámetros
- **Archivo de imagen**: Modifica `imagen_archivo = "pita.jpg"` con tu archivo
- **Longitud real**: Cambia `longitud_real_esperada = 57` por la longitud real de tu cuerda

## 🧮 Método Matemático

### Fórmula de Longitud de Arco
```
L = ∫[a,b] √(1 + (dy/dx)²) dx
```

### Proceso de Cálculo
1. **Extracción de puntos** de la imagen
2. **Traslación al origen** (0,0)
3. **Interpolación** con splines cúbicos
4. **Cálculo de derivadas** numéricas
5. **Aplicación del trapecio compuesto** con 500 subdivisiones
6. **Calibración** con longitud real conocida

## 📊 Resultados Generados

### Archivos CSV
- `funciones_polinomicas_detalladas.csv`: Funciones por segmentos
- `puntos_interpolacion.csv`: Coordenadas de todos los puntos

### Visualizaciones (12 gráficos)
1. **Curva original** con puntos extraídos
2. **Curva en centímetros** calibrada
3. **Derivada dy/dx** a lo largo de la curva
4. **Integrando** √(1 + (dy/dx)²)
5. **Convergencia** del método trapecio
6. **Funciones polinómicas locales**
7. **Distribución de pendientes**
8. **Mapa de pendientes**
9. **Comparación de escalas**
10. **Error de aproximación**
11. **Curvatura aproximada**
12. **Resumen numérico**

## 🔧 Documentación del Código

### Estructura Principal

```python
# === IMPORTACIONES ===
import numpy as np              # Cálculos numéricos y arrays
import matplotlib.pyplot as plt # Generación de gráficos
from scipy import interpolate   # Interpolación matemática
import cv2                      # Procesamiento de imágenes
from PIL import Image          # Manipulación adicional de imágenes
import os                      # Operaciones del sistema operativo
import pandas as pd            # Manejo de datos tabulares
```

### Funciones Principales

#### `procesar_imagen_cuerda(imagen_path)`
**Propósito**: Extrae automáticamente los puntos de la cuerda desde la imagen

**Proceso detallado**:
```python
def procesar_imagen_cuerda(imagen_path):
    # Verificar si el archivo existe
    if not os.path.exists(imagen_path):
        return None, None
    
    # Cargar la imagen en formato BGR (Blue-Green-Red)
    img = cv2.imread(imagen_path)
    
    # Convertir a escala de grises para mejor procesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Gaussiano para suavizar y reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes usando algoritmo Canny
    # Parámetros: imagen, umbral_bajo=20, umbral_alto=80
    edges = cv2.Canny(blurred, 20, 80)
    
    # Encontrar contornos de los bordes detectados
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filtrar contornos muy pequeños (menos de 50 puntos)
    contornos_largos = [c for c in contours if len(c) > 50]
    
    # Seleccionar el contorno más largo (probablemente la cuerda)
    largest_contour = max(contornos_largos, key=len)
    
    # Extraer coordenadas (x,y) del contorno
    points = largest_contour.reshape(-1, 2)
    x_pixels = points[:, 0].astype(float)  # Coordenadas X
    y_pixels = points[:, 1].astype(float)  # Coordenadas Y
    
    # Convertir coordenadas de imagen a matemáticas (invertir Y)
    h_img = img.shape[0]  # Altura de la imagen
    y_pixels = h_img - y_pixels  # Invertir eje Y
    
    # Ordenar puntos por coordenada X (de izquierda a derecha)
    sorted_indices = np.argsort(x_pixels)
    x_pixels = x_pixels[sorted_indices]
    y_pixels = y_pixels[sorted_indices]
    
    # Reducir a 25 puntos para interpolación más suave
    if len(x_pixels) > 25:
        indices = np.linspace(0, len(x_pixels)-1, 25, dtype=int)
        x_pixels = x_pixels[indices]
        y_pixels = y_pixels[indices]
    
    return x_pixels, y_pixels
```

#### `extraer_puntos_manual(img)`
**Propósito**: Permite selección manual de puntos cuando falla la detección automática

```python
def extraer_puntos_manual(img):
    # Crear figura de matplotlib para mostrar la imagen
    plt.figure(figsize=(12, 8))
    
    # Mostrar imagen (convertir de BGR a RGB para matplotlib)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    puntos = []  # Lista para almacenar puntos seleccionados
    
    def onclick(event):
        """Función que se ejecuta al hacer clic en la imagen"""
        if event.inaxes:  # Si el clic está dentro de los ejes
            x_click = event.xdata  # Coordenada X del clic
            y_click = event.ydata  # Coordenada Y del clic
            puntos.append((x_click, y_click))  # Guardar punto
            
            # Dibujar punto rojo en la posición
            plt.plot(x_click, y_click, 'ro', markersize=8)
            plt.draw()  # Actualizar gráfico
    
    # Conectar función onclick al evento de clic del mouse
    plt.connect('button_press_event', onclick)
    plt.show()  # Mostrar ventana interactiva
    
    # Procesar puntos seleccionados
    if puntos:
        x_coords = np.array([p[0] for p in puntos])  # Extraer coordenadas X
        y_coords = np.array([img.shape[0] - p[1] for p in puntos])  # Invertir Y
        return x_coords, y_coords
    
    return None, None
```

#### `trasladar_a_origen(x_points, y_points)`
**Propósito**: Mueve la curva para que inicie en el punto (0,0)

```python
def trasladar_a_origen(x_points, y_points):
    # Encontrar el punto con menor coordenada X
    x_min_idx = np.argmin(x_points)
    x_inicial = x_points[x_min_idx]  # Coordenada X inicial
    y_inicial = y_points[x_min_idx]  # Coordenada Y inicial
    
    # Trasladar todos los puntos restando las coordenadas iniciales
    x_trasladado = x_points - x_inicial  # Nuevo origen X = 0
    y_trasladado = y_points - y_inicial  # Nuevo origen Y = 0
    
    return x_trasladado, y_trasladado
```

#### `metodo_trapecio_compuesto(func, a, b, n)`
**Propósito**: Implementa el método numérico para calcular la longitud de arco

```python
def metodo_trapecio_compuesto(func, a, b, n):
    """
    Calcula: L = ∫[a,b] √(1 + (dy/dx)²) dx
    usando la regla del trapecio compuesto
    """
    
    h = (b - a) / n  # Ancho de cada subintervalo
    x = np.linspace(a, b, n + 1)  # n+1 puntos de evaluación
    
    # Evaluar la función spline en todos los puntos
    y = func(x)
    
    # Calcular derivada numérica dy/dx
    dy_dx = np.gradient(y, h)
    
    # Calcular el integrando: √(1 + (dy/dx)²)
    integrand = np.sqrt(1 + dy_dx**2)
    
    # Aplicar fórmula del trapecio compuesto:
    # ∫f(x)dx ≈ h/2 * (f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ))
    longitud = h * (integrand[0]/2 + np.sum(integrand[1:-1]) + integrand[-1]/2)
    
    return longitud, x, integrand, dy_dx
```

#### `interpolar_y_calcular_longitud(x_points, y_points, longitud_real_cm)`
**Propósito**: Función principal que coordina todo el proceso de cálculo

```python
def interpolar_y_calcular_longitud(x_points, y_points, longitud_real_cm=57):
    # 1. VALIDACIÓN DE DATOS
    if x_points is None or len(x_points) < 2:
        return None, None
    
    # 2. TRASLACIÓN AL ORIGEN
    x_points, y_points = trasladar_a_origen(x_points, y_points)
    
    # 3. ELIMINACIÓN DE DUPLICADOS
    # Mantener solo puntos con diferencia mínima de 1 pixel en X
    unique_indices = []
    prev_x = None
    for i, x in enumerate(x_points):
        if prev_x is None or abs(x - prev_x) > 1:
            unique_indices.append(i)
            prev_x = x
    
    x_points = x_points[unique_indices]
    y_points = y_points[unique_indices]
    
    # 4. SELECCIÓN DE MÉTODO DE INTERPOLACIÓN
    if len(x_points) >= 4:
        # Usar spline cúbico para suavizado superior
        spline = interpolate.CubicSpline(x_points, y_points)
        metodo = "Spline Cúbico"
    else:
        # Usar interpolación lineal para pocos puntos
        spline = interpolate.interp1d(x_points, y_points, kind='linear')
        metodo = "Lineal"
    
    # 5. CÁLCULO CON DIFERENTES SUBDIVISIONES
    x_min, x_max = x_points.min(), x_points.max()
    n_valores = [10, 20, 50, 100, 200, 500]  # Número de subdivisiones
    longitudes = []
    
    # Probar convergencia con diferentes subdivisiones
    for n in n_valores:
        L, _, _, _ = metodo_trapecio_compuesto(spline, x_min, x_max, n)
        longitudes.append(L)
    
    # 6. RESULTADO FINAL CON MÁXIMA PRECISIÓN
    L_final_pixels, x_calc, integrand_final, dy_dx_final = metodo_trapecio_compuesto(
        spline, x_min, x_max, 500
    )
    
    # 7. CALIBRACIÓN CON LONGITUD REAL
    factor_escala = longitud_real_cm / L_final_pixels  # cm/pixel
    L_final_cm = L_final_pixels * factor_escala
    
    # 8. GENERAR FUNCIONES POLINÓMICAS Y VISUALIZACIONES
    funciones_data = extraer_funciones_polinomicas_detalladas(
        spline, x_min, x_max, factor_escala, n_segmentos=12
    )
    
    generar_graficos_completos(
        x_points, y_points, spline, factor_escala, 
        funciones_data, L_final_pixels, metodo,
        x_calc, integrand_final, dy_dx_final, n_valores, longitudes
    )
    
    return L_final_cm, factor_escala
```

#### `extraer_funciones_polinomicas_detalladas(spline, x_min, x_max, factor_escala, n_segmentos)`
**Propósito**: Divide la curva en segmentos y calcula funciones polinómicas locales

```python
def extraer_funciones_polinomicas_detalladas(spline, x_min, x_max, factor_escala, n_segmentos=15):
    # Convertir límites a centímetros
    x_min_cm = x_min * factor_escala
    x_max_cm = x_max * factor_escala
    
    # Crear puntos de división para los segmentos
    x_segmentos_cm = np.linspace(x_min_cm, x_max_cm, n_segmentos + 1)
    
    funciones_data = []
    
    for i in range(n_segmentos):
        # Límites del segmento actual
        x_inicio_cm = x_segmentos_cm[i]
        x_fin_cm = x_segmentos_cm[i + 1]
        x_medio_cm = (x_inicio_cm + x_fin_cm) / 2
        
        # Convertir punto medio a pixels para evaluar el spline
        x_medio_pixels = x_medio_cm / factor_escala
        
        # Evaluar función y su derivada en el punto medio
        y_medio_pixels = float(spline(x_medio_pixels))      # f(x)
        dy_dx = float(spline(x_medio_pixels, 1))            # f'(x)
        
        # Convertir a centímetros
        y_medio_cm = y_medio_pixels * factor_escala
        
        # Crear función polinómica local (aproximación lineal)
        # Forma: y = y₀ + m×(x - x₀)
        funcion_str = f"y = {y_medio_cm:.4f} + {dy_dx:.4f}×(x - {x_medio_cm:.4f})"
        
        # Almacenar información del segmento
        funciones_data.append({
            'Segmento': i + 1,
            'Punto_Inicio': f"({x_inicio_cm:.3f}, {y_inicio_cm:.3f})",
            'Punto_Medio': f"({x_medio_cm:.3f}, {y_medio_cm:.3f})",
            'Punto_Final': f"({x_fin_cm:.3f}, {y_fin_cm:.3f})",
            'x_inicio_cm': round(x_inicio_cm, 4),
            'y_inicio_cm': round(y_inicio_cm, 4),
            'x_medio_cm': round(x_medio_cm, 4),
            'y_medio_cm': round(y_medio_cm, 4),
            'x_fin_cm': round(x_fin_cm, 4),
            'y_fin_cm': round(y_fin_cm, 4),
            'Pendiente_dy_dx': round(dy_dx, 4),
            'Funcion_Polinomica': funcion_str
        })
    
    return funciones_data
```

#### `generar_graficos_completos(...)`
**Propósito**: Crea las 12 visualizaciones del análisis

```python
def generar_graficos_completos(x_points, y_points, spline, factor_escala, 
                              funciones_data, L_final_pixels, metodo,
                              x_calc, integrand_final, dy_dx_final, n_valores, longitudes):
    
    # Crear figura con 12 subgráficos (3 filas × 4 columnas)
    plt.figure(figsize=(20, 15))
    
    # Preparar datos para visualización
    x_min, x_max = x_points.min(), x_points.max()
    x_interp = np.linspace(x_min, x_max, 300)  # 300 puntos para curva suave
    y_interp = spline(x_interp)                # Evaluar spline
    
    # Convertir a centímetros
    x_interp_cm = x_interp * factor_escala
    y_interp_cm = y_interp * factor_escala
    x_points_cm = x_points * factor_escala
    y_points_cm = y_points * factor_escala
    
    # GRÁFICO 1: Curva original en pixels
    plt.subplot(3, 4, 1)
    plt.plot(x_points, y_points, 'ro', markersize=8, label='Puntos Extraídos')
    plt.plot(x_interp, y_interp, 'b-', linewidth=3, label=f'{metodo}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Curva Original (Pixels)')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    
    # GRÁFICO 2: Curva calibrada en centímetros
    plt.subplot(3, 4, 2)
    plt.plot(x_points_cm, y_points_cm, 'ro', markersize=8, label='Puntos (cm)')
    plt.plot(x_interp_cm, y_interp_cm, 'g-', linewidth=3, label='Curva (cm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Curva en Centímetros')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    
    # GRÁFICO 3: Derivada dy/dx
    plt.subplot(3, 4, 3)
    plt.plot(x_calc, dy_dx_final, 'purple', linewidth=2, label='dy/dx')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Derivada dy/dx')
    plt.xlabel('x (pixels)')
    plt.ylabel('dy/dx')
    
    # GRÁFICO 4: Integrando para longitud de arco
    plt.subplot(3, 4, 4)
    plt.plot(x_calc, integrand_final, 'red', linewidth=2, label='√(1 + (dy/dx)²)')
    plt.fill_between(x_calc, integrand_final, alpha=0.3, color='red')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Integrando √(1 + (dy/dx)²)')
    plt.xlabel('x (pixels)')
    plt.ylabel('√(1 + (dy/dx)²)')
    
    # [Continúa con los otros 8 gráficos...]
    
    plt.tight_layout()  # Ajustar espaciado automáticamente
    plt.show()         # Mostrar todos los gráficos
```

#### `main()`
**Propósito**: Función principal que coordina todo el flujo del programa

```python
def main():
    # CONFIGURACIÓN INICIAL
    imagen_archivo = "pita.jpg"     # CAMBIA ESTE NOMBRE por tu imagen
    longitud_real_esperada = 57     # CAMBIA ESTE VALOR por tu longitud real (cm)
    
    try:
        # PASO 1: PROCESAMIENTO DE IMAGEN
        x_points, y_points = procesar_imagen_cuerda(imagen_archivo)
        
        if x_points is not None and y_points is not None:
            # PASO 2: CÁLCULO DE LONGITUD
            longitud_final, factor_escala = interpolar_y_calcular_longitud(
                x_points, y_points, longitud_real_cm=longitud_real_esperada
            )
            
            if longitud_final is not None:
                # MOSTRAR RESULTADOS FINALES
                print(f"🎯 LONGITUD TOTAL: {longitud_final:.2f} cm")
                print(f"📏 FACTOR DE ESCALA: {factor_escala:.6f} cm/pixel")
        
        else:
            # OFRECER MODO MANUAL si falla la detección automática
            respuesta = input("¿Deseas intentar el modo manual? (s/n): ")
            if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                img = cv2.imread(imagen_archivo)
                if img is not None:
                    x_manual, y_manual = extraer_puntos_manual(img)
                    if x_manual is not None and y_manual is not None:
                        longitud_manual, factor_manual = interpolar_y_calcular_longitud(
                            x_manual, y_manual, longitud_real_cm=longitud_real_esperada
                        )
    
    except FileNotFoundError:
        # MANEJO DE ERRORES: archivo no encontrado
        print(f"❌ ARCHIVO NO ENCONTRADO: '{imagen_archivo}'")
        
        # Buscar automáticamente archivos de imagen en el directorio
        archivos_imagen = []
        extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        for archivo in os.listdir('.'):
            if any(archivo.lower().endswith(ext) for ext in extensiones):
                archivos_imagen.append(archivo)
        
        # Mostrar archivos encontrados y permitir selección
        if archivos_imagen:
            print("📁 Archivos de imagen encontrados:")
            for i, archivo in enumerate(archivos_imagen, 1):
                print(f"   {i}. {archivo}")
```

### Funciones Auxiliares

#### `verificar_requisitos()`
**Propósito**: Verifica que todas las librerías estén instaladas correctamente

```python
def verificar_requisitos():
    requisitos = {
        'numpy': 'Cálculos numéricos',
        'matplotlib': 'Generación de gráficos',
        'scipy': 'Interpolación y métodos numéricos',
        'cv2': 'Procesamiento de imágenes (OpenCV)',
        'PIL': 'Manipulación de imágenes',
        'pandas': 'Manejo de datos tabulares'
    }
    
    faltantes = []
    
    # Intentar importar cada librería
    for libreria, descripcion in requisitos.items():
        try:
            if libreria == 'cv2':
                import cv2
            elif libreria == 'PIL':
                from PIL import Image
            else:
                __import__(libreria)
            print(f"✅ {libreria:12} - {descripcion}")
        except ImportError:
            print(f"❌ {libreria:12} - {descripcion} (FALTANTE)")
            faltantes.append(libreria)
    
    return len(faltantes) == 0  # True si no faltan librerías
```

#### `mostrar_ayuda()`
**Propósito**: Muestra información detallada sobre el uso del programa

```python
def mostrar_ayuda():
    print("📚 === AYUDA - CALCULADORA DE LONGITUD DE CUERDA ===")
    print("🎯 OBJETIVO:")
    print("   Calcular la longitud real de una cuerda a partir de una fotografía")
    print("   usando el método matemático de trapecio compuesto.")
    
    print("📋 REQUISITOS:")
    print("   1. Imagen clara de la cuerda con buen contraste")
    print("   2. Longitud real conocida de la cuerda (para calibración)")
    print("   3. Librerías: opencv-python, matplotlib, scipy, pandas, numpy")
    
    # [Continúa con más información de ayuda...]
```

## 🔬 Detalles Técnicos

### Algoritmo de Detección de Bordes Canny
1. **Filtro Gaussiano**: Suaviza la imagen para reducir ruido
2. **Gradiente**: Calcula la intensidad y dirección de los bordes
3. **Supresión no-máxima**: Adelgaza los bordes a 1 pixel
4. **Umbralización doble**: Distingue bordes fuertes de débiles
5. **Conectividad**: Une bordes débiles conectados a fuertes

### Interpolación con Splines Cúbicos
- **Continuidad C²**: Función, primera y segunda derivada continuas
- **Suavidad**: Minimiza la curvatura total
- **Precisión**: Pasa exactamente por todos los puntos de control

### Método de Trapecio Compuesto
- **Subdivisión**: Divide el intervalo [a,b] en n partes iguales
- **Aproximación**: Aproxima cada subintervalo con un trapecio
- **Convergencia**: Error decrece como O(h²) donde h = (b-a)/n

## 📁 Estructura de Archivos Generados

### `funciones_polinomicas_detalladas.csv`
```
Segmento,Punto_Inicio,Punto_Medio,Punto_Final,x_inicio_cm,y_inicio_cm,...
1,"(0.000, 0.000)","(2.375, 1.234)","(4.750, 2.468)",0.0000,0.0000,...
2,"(4.750, 2.468)","(7.125, 3.702)","(9.500, 4.936)",4.7500,2.4680,...
...
```

### `puntos_interpolacion.csv`
```
Punto,Coordenadas_Pixels,Coordenadas_CM,x_cm,y_cm
1,"(0.0, 0.0)","(0.000, 0.000)",0.0000,0.0000
2,"(15.2, 8.7)","(1.824, 1.044)",1.8240,1.0440
...
```

## 🐛 Solución de Problemas

### Error: "No se pudieron extraer puntos"
**Causa**: Detección automática falló
**Solución**: Usar modo manual o mejorar la imagen

### Error: "Archivo no encontrado"
**Causa**: Nombre de archivo incorrecto o ubicación
**Solución**: Verificar nombre y ubicación del archivo

### Error: "ImportError"
**Causa**: Librerías faltantes
**Solución**: Ejecutar `pip install [libreria_faltante]`

### Resultados imprecisos
**Causa**: Imagen de baja calidad o mala calibración
**Solución**: 
- Usar imagen con mejor resolución y contraste
- Verificar la longitud real de calibración
- Asegurar que la cuerda esté completamente estirada

## 👥 Colaboradores

Este README fue creado para facilitar el uso del programa por parte del equipo de trabajo. Si tienes dudas o sugerencias, no dudes en preguntar.

## 📚 Referencias Matemáticas

- **Método de trapecio compuesto**: Burden, R. & Faires, J. "Numerical Analysis" - Para la implementación del método numérico de integración
- **Splines cúbicos**: De Boor, C. "A Practical Guide to Splines" - Para la interpolación suave de la curva
- **Procesamiento de imágenes**: Gonzalez, R. "Digital Image Processing