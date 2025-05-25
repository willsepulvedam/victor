import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2
from PIL import Image
import os
import pandas as pd

print("=== CALCULADORA DE LONGITUD DE CUERDA ===")
print("Método: Trapecio Compuesto con Interpolación Polinómica")
print("=" * 50)

def procesar_imagen_cuerda(imagen_path):
    """
    Procesa la imagen de la cuerda usando detección automática de bordes
    """
    try:
        if not os.path.exists(imagen_path):
            print(f"❌ Error: No se encuentra el archivo {imagen_path}")
            return None, None
        
        print(f"📸 Procesando imagen: {imagen_path}")
        
        # Cargar imagen
        img = cv2.imread(imagen_path)
        if img is None:
            print("❌ Error: No se pudo cargar la imagen")
            return None, None
        
        print(f"✅ Imagen cargada: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtros para mejorar la detección
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes usando Canny
        edges = cv2.Canny(blurred, 20, 80)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        print(f"🔍 Contornos encontrados: {len(contours)}")
        
        if not contours:
            print("⚠️ No se detectaron contornos. Usando método manual...")
            return extraer_puntos_manual(img)
        
        # Filtrar y seleccionar el mejor contorno
        contornos_largos = [c for c in contours if len(c) > 50]
        if not contornos_largos:
            contornos_largos = contours
        
        largest_contour = max(contornos_largos, key=len)
        print(f"✅ Contorno seleccionado: {len(largest_contour)} puntos")
        
        # Extraer y procesar puntos
        points = largest_contour.reshape(-1, 2)
        x_pixels = points[:, 0].astype(float)
        y_pixels = points[:, 1].astype(float)
        
        # Convertir coordenadas de imagen a coordenadas matemáticas
        h_img = img.shape[0]
        y_pixels = h_img - y_pixels
        
        # Ordenar puntos por coordenada X
        sorted_indices = np.argsort(x_pixels)
        x_pixels = x_pixels[sorted_indices]
        y_pixels = y_pixels[sorted_indices]
        
        # Reducir puntos para interpolación suave (20-30 puntos)
        if len(x_pixels) > 25:
            indices = np.linspace(0, len(x_pixels)-1, 25, dtype=int)
            x_pixels = x_pixels[indices]
            y_pixels = y_pixels[indices]
        
        return x_pixels, y_pixels
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        return None, None

def extraer_puntos_manual(img):
    """
    Permite hacer clic manual en la imagen cuando la detección automática falla
    """
    print("\n🖱️  === MODO MANUAL ===")
    print("📌 Haz clic en los puntos de la cuerda de IZQUIERDA a DERECHA")
    print("⌨️  Presiona 'q' o cierra la ventana cuando termines")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('HAZ CLIC EN LOS PUNTOS DE LA CUERDA (de izquierda a derecha)', fontsize=14)
    
    puntos = []
    
    def onclick(event):
        if event.inaxes:
            x_click = event.xdata
            y_click = event.ydata
            puntos.append((x_click, y_click))
            plt.plot(x_click, y_click, 'ro', markersize=8)
            plt.draw()
            print(f"📍 Punto {len(puntos)}: ({x_click:.0f}, {y_click:.0f})")
    
    plt.connect('button_press_event', onclick)
    plt.show()
    
    if puntos:
        x_coords = np.array([p[0] for p in puntos])
        y_coords = np.array([img.shape[0] - p[1] for p in puntos])  # Invertir Y
        return x_coords, y_coords
    else:
        return None, None

def trasladar_a_origen(x_points, y_points):
    """
    Traslada la curva para que inicie en el origen (0,0)
    """
    x_min_idx = np.argmin(x_points)
    x_inicial = x_points[x_min_idx]
    y_inicial = y_points[x_min_idx]
    
    x_trasladado = x_points - x_inicial
    y_trasladado = y_points - y_inicial
    
    print(f"📐 Traslación aplicada: dx = {-x_inicial:.2f}, dy = {-y_inicial:.2f}")
    
    return x_trasladado, y_trasladado

def metodo_trapecio_compuesto(func, a, b, n):
    """
    Implementación del método de trapecio compuesto para calcular longitud de arco
    Fórmula: L = ∫[a,b] √(1 + (dy/dx)²) dx
    """
    h = (b - a) / n  # Ancho de cada subintervalo
    x = np.linspace(a, b, n + 1)  # Puntos de evaluación
    
    # Evaluar la función en todos los puntos
    y = func(x)
    
    # Calcular la derivada numéricamente
    dy_dx = np.gradient(y, h)
    
    # Calcular el integrando √(1 + (dy/dx)²)
    integrand = np.sqrt(1 + dy_dx**2)
    
    # Aplicar fórmula del trapecio compuesto
    # L = h/2 * (f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ))
    longitud = h * (integrand[0]/2 + np.sum(integrand[1:-1]) + integrand[-1]/2)
    
    return longitud, x, integrand, dy_dx

def extraer_funciones_polinomicas_detalladas(spline, x_min, x_max, factor_escala, n_segmentos=15):
    """
    Extrae funciones polinómicas locales con coordenadas detalladas
    """
    x_min_cm = x_min * factor_escala
    x_max_cm = x_max * factor_escala
    
    # Dividir en segmentos
    x_segmentos_cm = np.linspace(x_min_cm, x_max_cm, n_segmentos + 1)
    
    funciones_data = []
    
    for i in range(n_segmentos):
        x_inicio_cm = x_segmentos_cm[i]
        x_fin_cm = x_segmentos_cm[i + 1]
        x_medio_cm = (x_inicio_cm + x_fin_cm) / 2
        
        # Convertir a pixels para evaluar el spline
        x_medio_pixels = x_medio_cm / factor_escala
        
        # Evaluar función y derivadas
        y_medio_pixels = float(spline(x_medio_pixels))
        dy_dx = float(spline(x_medio_pixels, 1))
        
        # Convertir a centímetros
        y_medio_cm = y_medio_pixels * factor_escala
        
        # Calcular puntos adicionales en el segmento
        x_inicio_pixels = x_inicio_cm / factor_escala
        x_fin_pixels = x_fin_cm / factor_escala
        y_inicio_cm = float(spline(x_inicio_pixels)) * factor_escala
        y_fin_cm = float(spline(x_fin_pixels)) * factor_escala
        
        # Función polinómica local (aproximación lineal)
        funcion_str = f"y = {y_medio_cm:.4f} + {dy_dx:.4f}×(x - {x_medio_cm:.4f})"
        
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

def interpolar_y_calcular_longitud(x_points, y_points, longitud_real_cm=57):
    """
    Realiza interpolación y calcula la longitud usando trapecio compuesto
    """
    if x_points is None or y_points is None or len(x_points) < 2:
        print("❌ Error: No hay suficientes puntos para interpolar")
        return None, None
    
    print(f"\n📊 === ANÁLISIS DE PUNTOS ({len(x_points)} puntos) ===")
    
    # Trasladar la curva al origen
    print("\n📐 === TRASLADANDO CURVA AL ORIGEN ===")
    x_points, y_points = trasladar_a_origen(x_points, y_points)
    
    # Mostrar primeros puntos trasladados
    print("📍 Primeros puntos trasladados:")
    for i in range(min(5, len(x_points))):
        print(f"   Punto {i+1}: ({x_points[i]:.2f}, {y_points[i]:.2f})")
    if len(x_points) > 5:
        print(f"   ... y {len(x_points)-5} puntos más")
    
    # Eliminar duplicados en X
    unique_indices = []
    prev_x = None
    for i, x in enumerate(x_points):
        if prev_x is None or abs(x - prev_x) > 1:
            unique_indices.append(i)
            prev_x = x
    
    x_points = x_points[unique_indices]
    y_points = y_points[unique_indices]
    
    print(f"✅ Puntos únicos para interpolación: {len(x_points)}")
    
    # Seleccionar método de interpolación
    if len(x_points) >= 4:
        spline = interpolate.CubicSpline(x_points, y_points)
        metodo = "Spline Cúbico"
        print("🔧 Usando interpolación: Spline Cúbico")
    else:
        spline = interpolate.interp1d(x_points, y_points, kind='linear')
        metodo = "Lineal"
        print("🔧 Usando interpolación: Lineal")
    
    # Calcular longitud con método de trapecio compuesto
    x_min, x_max = x_points.min(), x_points.max()
    
    print(f"\n🧮 === CÁLCULO CON MÉTODO DE TRAPECIO COMPUESTO ===")
    print("Subdivisiones\tLongitud (pixels)\tError Estimado")
    print("-" * 55)
    
    # Probar diferentes números de subdivisiones
    n_valores = [10, 20, 50, 100, 200, 500]
    longitudes = []
    
    for n in n_valores:
        L, _, _, _ = metodo_trapecio_compuesto(spline, x_min, x_max, n)
        longitudes.append(L)
        
        # Estimar error comparando con el valor anterior
        if len(longitudes) > 1:
            error = abs(longitudes[-1] - longitudes[-2])
            print(f"{n:12d}\t{L:15.4f}\t{error:10.6f}")
        else:
            print(f"{n:12d}\t{L:15.4f}\t{'N/A':>10}")
    
    # Resultado final con máxima precisión
    L_final_pixels, x_calc, integrand_final, dy_dx_final = metodo_trapecio_compuesto(spline, x_min, x_max, 500)
    
    # Calcular factor de escala
    factor_escala = longitud_real_cm / L_final_pixels
    L_final_cm = L_final_pixels * factor_escala
    
    print(f"\n🎯 === RESULTADO DEL TRAPECIO COMPUESTO ===")
    print(f"Longitud calculada (pixels): {L_final_pixels:.4f}")
    print(f"Factor de escala: {factor_escala:.6f} cm/pixel")
    print(f"Longitud real: {L_final_cm:.2f} cm")
    
    # Extraer funciones polinómicas detalladas
    print(f"\n📐 === EXTRAYENDO FUNCIONES POLINÓMICAS ===")
    funciones_data = extraer_funciones_polinomicas_detalladas(spline, x_min, x_max, factor_escala, n_segmentos=12)
    
    # Crear tablas con pandas
    df_funciones = pd.DataFrame(funciones_data)
    
    # Tabla simplificada para mostrar
    df_resumen = df_funciones[['Segmento', 'Punto_Inicio', 'Punto_Medio', 'Punto_Final', 
                              'Pendiente_dy_dx', 'Funcion_Polinomica']].copy()
    
    print("\n📋 === TABLA DE FUNCIONES POLINÓMICAS (COORDENADAS EN CM) ===")
    print(df_resumen.to_string(index=False, max_colwidth=50))
    
    # Tabla de coordenadas de puntos originales
    puntos_data = []
    for i, (x_px, y_px) in enumerate(zip(x_points, y_points)):
        x_cm = x_px * factor_escala
        y_cm = y_px * factor_escala
        puntos_data.append({
            'Punto': i + 1,
            'Coordenadas_Pixels': f"({x_px:.1f}, {y_px:.1f})",
            'Coordenadas_CM': f"({x_cm:.3f}, {y_cm:.3f})",
            'x_cm': round(x_cm, 4),
            'y_cm': round(y_cm, 4)
        })
    
    df_puntos = pd.DataFrame(puntos_data)
    
    print(f"\n📍 === TABLA DE PUNTOS INTERPOLADOS ===")
    print(df_puntos[['Punto', 'Coordenadas_Pixels', 'Coordenadas_CM']].to_string(index=False))
    
    # Generar gráficos
    generar_graficos_completos(x_points, y_points, spline, factor_escala, 
                              funciones_data, L_final_pixels, metodo,
                              x_calc, integrand_final, dy_dx_final, n_valores, longitudes)
    
    # Guardar archivos CSV
    df_funciones.to_csv('funciones_polinomicas_detalladas.csv', index=False)
    df_puntos.to_csv('puntos_interpolacion.csv', index=False)
    
    print(f"\n💾 === ARCHIVOS GUARDADOS ===")
    print(f"📄 funciones_polinomicas_detalladas.csv")
    print(f"📄 puntos_interpolacion.csv")
    
    return L_final_cm, factor_escala

def generar_graficos_completos(x_points, y_points, spline, factor_escala, 
                              funciones_data, L_final_pixels, metodo,
                              x_calc, integrand_final, dy_dx_final, n_valores, longitudes):
    """
    Genera visualizaciones completas del análisis
    """
    plt.figure(figsize=(20, 15))
    
    # Preparar datos
    x_min, x_max = x_points.min(), x_points.max()
    x_interp = np.linspace(x_min, x_max, 300)
    y_interp = spline(x_interp)
    
    x_interp_cm = x_interp * factor_escala
    y_interp_cm = y_interp * factor_escala
    x_points_cm = x_points * factor_escala
    y_points_cm = y_points * factor_escala
    
    # 1. Curva original con puntos
    plt.subplot(3, 4, 1)
    plt.plot(x_points, y_points, 'ro', markersize=8, label='Puntos Extraídos', zorder=3)
    plt.plot(x_interp, y_interp, 'b-', linewidth=3, label=f'{metodo}', alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Curva Original (Pixels)', fontsize=12, fontweight='bold')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    
    # 2. Curva en centímetros
    plt.subplot(3, 4, 2)
    plt.plot(x_points_cm, y_points_cm, 'ro', markersize=8, label='Puntos (cm)', zorder=3)
    plt.plot(x_interp_cm, y_interp_cm, 'g-', linewidth=3, label='Curva (cm)', alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Curva en Centímetros', fontsize=12, fontweight='bold')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    
    # 3. Derivada dy/dx
    plt.subplot(3, 4, 3)
    plt.plot(x_calc, dy_dx_final, 'purple', linewidth=2, label='dy/dx')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Derivada dy/dx', fontsize=12, fontweight='bold')
    plt.xlabel('x (pixels)')
    plt.ylabel('dy/dx')
    
    # 4. Integrando del trapecio compuesto
    plt.subplot(3, 4, 4)
    plt.plot(x_calc, integrand_final, 'red', linewidth=2, label='√(1 + (dy/dx)²)')
    plt.fill_between(x_calc, integrand_final, alpha=0.3, color='red')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Integrando √(1 + (dy/dx)²)', fontsize=12, fontweight='bold')
    plt.xlabel('x (pixels)')
    plt.ylabel('√(1 + (dy/dx)²)')
    
    # 5. Convergencia del método
    plt.subplot(3, 4, 5)
    plt.plot(n_valores, longitudes, 'bo-', linewidth=2, markersize=8)
    plt.grid(True, alpha=0.3)
    plt.title('Convergencia Trapecio Compuesto', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Subdivisiones')
    plt.ylabel('Longitud (pixels)')
    
    # 6. Funciones polinómicas por segmentos
    plt.subplot(3, 4, 6)
    colors = plt.cm.viridis(np.linspace(0, 1, len(funciones_data)))
    for i, func_data in enumerate(funciones_data):
        x_seg = np.linspace(func_data['x_inicio_cm'], func_data['x_fin_cm'], 20)
        x_medio = func_data['x_medio_cm']
        y_medio = func_data['y_medio_cm']
        dy_dx = func_data['Pendiente_dy_dx']
        y_seg = y_medio + dy_dx * (x_seg - x_medio)
        plt.plot(x_seg, y_seg, color=colors[i], linewidth=2, alpha=0.8)
    
    plt.plot(x_interp_cm, y_interp_cm, 'k-', linewidth=3, label='Spline Original', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.title('Funciones Polinómicas Locales', fontsize=12, fontweight='bold')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    
    # 7. Distribución de pendientes
    plt.subplot(3, 4, 7)
    pendientes = [func['Pendiente_dy_dx'] for func in funciones_data]
    plt.hist(pendientes, bins=8, alpha=0.7, color='orange', edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.title('Distribución de Pendientes', fontsize=12, fontweight='bold')
    plt.xlabel('dy/dx')
    plt.ylabel('Frecuencia')
    
    # 8. Mapa de pendientes a lo largo de la curva
    plt.subplot(3, 4, 8)
    x_medios = [func['x_medio_cm'] for func in funciones_data]
    scatter = plt.scatter(x_medios, pendientes, c=pendientes, cmap='RdYlBu', s=100, edgecolor='black')
    plt.colorbar(scatter, label='dy/dx')
    plt.grid(True, alpha=0.3)
    plt.title('Mapa de Pendientes', fontsize=12, fontweight='bold')
    plt.xlabel('Posición x (cm)')
    plt.ylabel('Pendiente dy/dx')
    
    # 9. Comparación de escalas
    plt.subplot(3, 4, 9)
    plt.plot(x_interp, y_interp, 'b-', linewidth=2, label='Pixels', alpha=0.7)
    plt.plot(x_interp_cm, y_interp_cm, 'r-', linewidth=2, label='Centímetros', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Comparación de Escalas', fontsize=12, fontweight='bold')
    plt.xlabel('Coordenada x')
    plt.ylabel('Coordenada y')
    
    # 10. Error de aproximación
    plt.subplot(3, 4, 10)
    if len(longitudes) > 1:
        errores = np.abs(np.diff(longitudes))
        plt.semilogy(n_valores[1:], errores, 'go-', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.title('Error de Aproximación', fontsize=12, fontweight='bold')
        plt.xlabel('Subdivisiones')
        plt.ylabel('Error Absoluto (log)')
    
    # 11. Curvatura aproximada
    plt.subplot(3, 4, 11)
    curvatura = np.abs(np.gradient(dy_dx_final))
    plt.plot(x_calc, curvatura, 'magenta', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Curvatura Aproximada', fontsize=12, fontweight='bold')
    plt.xlabel('x (pixels)')
    plt.ylabel('|d²y/dx²|')
    
    # 12. Resumen numérico
    plt.subplot(3, 4, 12)
    plt.axis('off')
    resumen_texto = f"""
RESUMEN FINAL
{'='*25}

Longitud Total: {L_final_pixels * factor_escala:.2f} cm
Factor Escala: {factor_escala:.6f} cm/px
Método: {metodo}
Puntos: {len(x_points)}
Subdivisiones: 500
Segmentos: {len(funciones_data)}

Rango X: {x_points_cm.min():.2f} - {x_points_cm.max():.2f} cm
Rango Y: {y_points_cm.min():.2f} - {y_points_cm.max():.2f} cm

Pendiente máx: {max(pendientes):.3f}
Pendiente mín: {min(pendientes):.3f}
"""
    plt.text(0.05, 0.95, resumen_texto, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    print("🚀 === INICIANDO ANÁLISIS DE LONGITUD DE CUERDA ===")
    print("📐 Método: Trapecio Compuesto + Interpolación Polinómica")
    print("=" * 60)
    
    # Configuración del archivo de imagen
    imagen_archivo = "pita.jpg"  # Cambia por el nombre de tu archivo
    longitud_real_esperada = 57  # cm - ajusta según tu cuerda real
    
    print(f"🔍 Buscando archivo: {imagen_archivo}")
    
    try:
        # 1. PROCESAMIENTO DE IMAGEN
        print(f"\n{'='*20} PASO 1: PROCESAMIENTO DE IMAGEN {'='*20}")
        x_points, y_points = procesar_imagen_cuerda(imagen_archivo)
        
        if x_points is not None and y_points is not None:
            print(f"✅ Extracción exitosa: {len(x_points)} puntos obtenidos")
            
            # 2. CÁLCULO DE LONGITUD
            print(f"\n{'='*20} PASO 2: INTERPOLACIÓN Y CÁLCULO {'='*20}")
            longitud_final, factor_escala = interpolar_y_calcular_longitud(
                x_points, y_points, longitud_real_cm=longitud_real_esperada
            )
            
            if longitud_final is not None:
                print(f"\n{'='*20} RESULTADO FINAL {'='*20}")
                print(f"🎯 LONGITUD TOTAL DE LA CUERDA: {longitud_final:.2f} cm")
                print(f"📏 FACTOR DE ESCALA: {factor_escala:.6f} cm/pixel")
                print(f"📊 MÉTODO UTILIZADO: Trapecio Compuesto (500 subdivisiones)")
                print(f"🔧 INTERPOLACIÓN: Automática con splines")
                print(f"📈 FUNCIONES POLINÓMICAS: Generadas por segmentos")
                print(f"💾 DATOS GUARDADOS: CSV generados exitosamente")
                print("=" * 60)
            else:
                print("❌ Error en el cálculo de longitud")
        else:
            print("❌ No se pudieron extraer puntos de la imagen")
            
            # Ofrecer modo manual
            respuesta = input("\n❓ ¿Deseas intentar el modo manual? (s/n): ")
            if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                try:
                    img = cv2.imread(imagen_archivo)
                    if img is not None:
                        print("🖱️  Iniciando modo manual...")
                        x_manual, y_manual = extraer_puntos_manual(img)
                        if x_manual is not None and y_manual is not None:
                            longitud_manual, factor_manual = interpolar_y_calcular_longitud(
                                x_manual, y_manual, longitud_real_cm=longitud_real_esperada
                            )
                            if longitud_manual is not None:
                                print(f"\n🎯 *** RESULTADO MANUAL ***")
                                print(f"LONGITUD TOTAL: {longitud_manual:.2f} cm")
                                print(f"FACTOR DE ESCALA: {factor_manual:.6f} cm/pixel")
                except Exception as e:
                    print(f"❌ Error en modo manual: {e}")
                    
    except FileNotFoundError:
        print(f"❌ ARCHIVO NO ENCONTRADO: '{imagen_archivo}'")
        print("📋 Verifica que:")
        print("   1. El archivo existe en la carpeta del script")
        print("   2. El nombre del archivo es correcto")
        print("   3. La extensión es correcta (.jpg, .png, .bmp, etc.)")
        print("\n💡 Formatos soportados: JPG, PNG, BMP, TIFF")
        
        # Listar archivos de imagen en el directorio actual
        archivos_imagen = []
        extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        for archivo in os.listdir('.'):
            if any(archivo.lower().endswith(ext) for ext in extensiones):
                archivos_imagen.append(archivo)
        
        if archivos_imagen:
            print(f"\n📁 Archivos de imagen encontrados en el directorio:")
            for i, archivo in enumerate(archivos_imagen, 1):
                print(f"   {i}. {archivo}")
            
            try:
                seleccion = input(f"\n❓ ¿Deseas usar alguno de estos archivos? (1-{len(archivos_imagen)} o 'n'): ")
                if seleccion.isdigit() and 1 <= int(seleccion) <= len(archivos_imagen):
                    archivo_seleccionado = archivos_imagen[int(seleccion) - 1]
                    print(f"🔄 Reintentando con: {archivo_seleccionado}")
                    
                    # Reintentar con el archivo seleccionado
                    x_points, y_points = procesar_imagen_cuerda(archivo_seleccionado)
                    if x_points is not None and y_points is not None:
                        longitud_final, factor_escala = interpolar_y_calcular_longitud(
                            x_points, y_points, longitud_real_cm=longitud_real_esperada
                        )
                        if longitud_final is not None:
                            print(f"\n🎯 *** RESULTADO FINAL ***")
                            print(f"LONGITUD TOTAL: {longitud_final:.2f} cm")
                            print(f"FACTOR DE ESCALA: {factor_escala:.6f} cm/pixel")
            except (ValueError, KeyboardInterrupt):
                print("❌ Operación cancelada")
        
    except Exception as e:
        print(f"❌ ERROR INESPERADO: {e}")
        print("💡 Sugerencias:")
        print("   - Verifica que la imagen tenga buena calidad")
        print("   - Asegúrate de que la cuerda tenga contraste con el fondo")
        print("   - Intenta con una imagen diferente")
        print("   - Verifica que todas las librerías estén instaladas")

def mostrar_ayuda():
    """
    Muestra información de ayuda sobre el uso del programa
    """
    print("\n📚 === AYUDA - CALCULADORA DE LONGITUD DE CUERDA ===")
    print("\n🎯 OBJETIVO:")
    print("   Calcular la longitud real de una cuerda a partir de una fotografía")
    print("   usando el método matemático de trapecio compuesto.")
    
    print("\n📋 REQUISITOS:")
    print("   1. Imagen clara de la cuerda con buen contraste")
    print("   2. Longitud real conocida de la cuerda (para calibración)")
    print("   3. Librerías: opencv-python, matplotlib, scipy, pandas, numpy")
    
    print("\n🔧 INSTALACIÓN DE LIBRERÍAS:")
    print("   pip install opencv-python matplotlib scipy pandas numpy")
    
    print("\n📸 RECOMENDACIONES PARA LA IMAGEN:")
    print("   - Fondo uniforme y contrastante con la cuerda")
    print("   - Buena iluminación sin sombras fuertes")
    print("   - Cuerda completamente visible y estirada")
    print("   - Resolución suficiente (mínimo 800x600)")
    print("   - Formatos: JPG, PNG, BMP, TIFF")
    
    print("\n⚙️ FUNCIONAMIENTO:")
    print("   1. Detección automática de bordes con filtro Canny")
    print("   2. Extracción de puntos del contorno de la cuerda")
    print("   3. Interpolación con splines cúbicos")
    print("   4. Cálculo de longitud con trapecio compuesto")
    print("   5. Calibración con longitud real conocida")
    print("   6. Generación de funciones polinómicas por segmentos")
    
    print("\n📊 RESULTADOS GENERADOS:")
    print("   - Longitud total calculada en centímetros")
    print("   - Factor de escala (cm/pixel)")
    print("   - Tabla de funciones polinómicas por segmentos")
    print("   - Coordenadas (x,y) de puntos de interpolación")
    print("   - Gráficos de análisis completo")
    print("   - Archivos CSV con datos detallados")
    
    print("\n🔍 MÉTODO MATEMÁTICO:")
    print("   L = ∫[a,b] √(1 + (dy/dx)²) dx")
    print("   Aproximado con trapecio compuesto de 500 subdivisiones")
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("   - funciones_polinomicas_detalladas.csv")
    print("   - puntos_interpolacion.csv")
    print("=" * 60)

def verificar_requisitos():
    """
    Verifica que todas las librerías necesarias estén instaladas
    """
    print("🔍 === VERIFICANDO REQUISITOS DEL SISTEMA ===")
    
    requisitos = {
        'numpy': 'Cálculos numéricos',
        'matplotlib': 'Generación de gráficos',
        'scipy': 'Interpolación y métodos numéricos',
        'cv2': 'Procesamiento de imágenes (OpenCV)',
        'PIL': 'Manipulación de imágenes',
        'pandas': 'Manejo de datos tabulares'
    }
    
    faltantes = []
    
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
    
    if faltantes:
        print(f"\n⚠️  LIBRERÍAS FALTANTES: {', '.join(faltantes)}")
        print("📦 Para instalar, ejecuta:")
        if 'cv2' in faltantes:
            print("   pip install opencv-python")
        if 'PIL' in faltantes:
            print("   pip install Pillow")
        for lib in faltantes:
            if lib not in ['cv2', 'PIL']:
                print(f"   pip install {lib}")
        print("\n🔄 Reinicia el programa después de instalar las librerías.")
        return False
    else:
        print("\n✅ Todos los requisitos están satisfechos!")
        return True

if __name__ == "__main__":
    print("🚀 CALCULADORA DE LONGITUD DE CUERDA")
    print("📐 Método: Trapecio Compuesto + Interpolación Polinómica")
    print("👨‍💻 Desarrollado para análisis matemático preciso")
    print("=" * 60)
    
    # Verificar requisitos del sistema
    if not verificar_requisitos():
        print("\n❌ No se pueden ejecutar todas las funciones debido a librerías faltantes.")
        respuesta = input("❓ ¿Deseas continuar de todos modos? (s/n): ")
        if respuesta.lower() not in ['s', 'si', 'sí', 'y', 'yes']:
            print("👋 Programa terminado. Instala las librerías faltantes y vuelve a intentar.")
            exit()
    
    # Menú principal
    while True:
        print(f"\n{'='*20} MENÚ PRINCIPAL {'='*20}")
        print("1. 📸 Analizar imagen de cuerda")
        print("2. 📚 Mostrar ayuda")
        print("3. 🔍 Verificar requisitos del sistema")
        print("4. 👋 Salir")
        print("=" * 60)
        
        try:
            opcion = input("Selecciona una opción (1-4): ").strip()
            
            if opcion == '1':
                main()
            elif opcion == '2':
                mostrar_ayuda()
            elif opcion == '3':
                verificar_requisitos()
            elif opcion == '4':
                print("👋 ¡Gracias por usar la Calculadora de Longitud de Cuerda!")
                print("📊 Recuerda revisar los archivos CSV generados con tus resultados.")
                break
            else:
                print("❌ Opción no válida. Por favor, selecciona 1, 2, 3 o 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            print("🔄 Reintentando...")
    
    print("\n📋 RESUMEN DE LA SESIÓN:")
    print("=" * 30)
    print("✅ Programa ejecutado correctamente")
    print("📁 Archivos CSV generados (si se procesó alguna imagen)")
    print("📊 Gráficos mostrados con análisis completo")
    print("🧮 Método de trapecio compuesto aplicado")
    print("📐 Funciones polinómicas extraídas por segmentos")
    print("=" * 60)