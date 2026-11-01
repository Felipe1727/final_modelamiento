# Sistema 5: tres masas, tres resortes, caja con rebotes

Proyecto final de Modelamiento Matemático (UPB). Tres masas en el plano xy, unidas por tres
resortes en triángulo (k1: m1–m2, k2: m2–m3, k3: m1–m3), con gravedad en −y, fricción
lineal del aire (−b·v) y rebotes contra las cuatro paredes de una caja. Las ecuaciones se
integran con un RK4 propio, sin scipy.

## Cómo ejecutar

Requisitos: Python 3.9 o superior, con `numpy`, `matplotlib` y `tkinter`.

```bash
python sistema5.py             # GUI con animación en vivo
python sistema5.py --validar   # pruebas A–H: escribe PASA/FALLA (~17 s)
```

Atajo en la GUI: la barra espaciadora inicia o pausa la simulación.

## Archivos

| Archivo | Contenido |
|---|---|
| `sistema5.py` | Todo el código, en un solo archivo para compartirlo fácil |
| `Sistema5_modelamiento.ipynb` | Desarrollo: física, solución analítica, discretización y validación con gráficas |

Secciones de `sistema5.py`:

1. **Modelo**: `RESORTES`, `Parametros` (con `h_sugerido`), `derivadas`, `energia`.
2. **Integrador**: `paso_rk4`, `paso_euler`, `aplicar_rebotes`, `penetracion`,
   `paso_con_eventos`, `simular`.
3. **Simulador**: guarda el estado actual. Métodos: `avanzar(n)`, `reiniciar(p)`,
   `actualizar_parametros(k, b, m, g, e)` y `fijar_paso(h)`.
4. **Analítica**: `oscilador_amortiguado`, `caso_respiracion`.
5. **Validación**: pruebas A–F del notebook, más G (el Simulador da lo mismo que `simular`
   y acepta cambios en vivo) y H (rechaza entradas inválidas).
6. **GUI**: `Control` (slider + campo numérico) y `App` (Tkinter + Matplotlib embebido).

Las funciones del modelo y del integrador son copia exacta de las del notebook (se verificó
comparando el AST). Solo se agregaron type hints y docstrings. Único cambio de
comportamiento: `Parametros` lanza `ValueError` en lugar de usar `assert` cuando una masa
empieza fuera de la caja.

## Decisiones de modelado

| Decisión | Elección | Por qué |
|---|---|---|
| Longitud natural L | Distancia inicial entre cada par; se recalcula al reiniciar | El triángulo arranca relajado y no agrega parámetros |
| Caja | [0, 10] × [0, 6] m, piso en y = 0 | La gravedad apunta al piso |
| Tamaño de las masas | Discos de radio R = 0.25 m | Hace falta un tamaño para detectar el choque |
| Rebote | Coeficiente de restitución e (e = 1: elástico); invierte solo la componente normal | El enunciado solo dice "rebotar" |
| Instante del choque | Se localiza por bisección dentro del paso (tolerancia 1e-9 s) | El rebote simple, al final del paso, deja un error de energía O(h) por choque |
| Contacto en reposo | Si \|v_n\| < V_REPOSO = 0.02 m/s, la masa queda apoyada (v_n = 0) | Evita la cascada infinita de micro-rebotes (paradoja de Zenón) |
| Choques entre masas | No se modelan | No se piden; las masas pueden atravesarse |
| Paso h | 1 ms por defecto; la GUI muestra h sugerido = T_min/100 | Con k ~ 10–100 N/m y m ~ 1 kg, 1 ms está muy dentro del límite de estabilidad |

### Limitaciones conocidas (para la defensa)

- **El contacto en reposo depende de h.** La masa queda quieta solo si g·h < V_REPOSO
  (h ≲ 2 ms con g = 9.81). Con h mayor aparecen micro-rebotes de ~10⁻⁵ m, invisibles en
  pantalla. La GUI avisa cuando pasa esto.
- **El contacto en reposo se aplica incluso con e = 1.** Una masa que roza una pared con
  \|v_n\| < 0.02 m/s pierde esa componente. La pérdida máxima es ½·m·0.02² ≈ 2·10⁻⁴ J,
  despreciable.
- **Masas que se cruzan.** Como no hay choques entre masas, dos masas pueden pasar una a
  través de la otra. Si L > 0, la fuerza −k(d − L)û cambia de dirección de golpe en d = 0
  (el potencial tiene una punta) y RK4 no resuelve bien ese instante. Con b = 0 y e = 1 se ve
  como un pequeño salto de energía. En el preset elástico, la energía se conserva mejor que
  10⁻⁶ durante 49 s; un cruce a 1 cm la deja en 2.6·10⁻⁵.

## La GUI

- **Pestaña "Física (en vivo)"**: k1–k3, m1–m3, b, g, e y h. Los cambios se aplican desde el
  siguiente paso, sin reiniciar; L no cambia. Si cambias k, m o g, la energía total salta
  porque el sistema pasa a ser otro. Cada cambio queda marcado con una línea punteada en el
  panel de energía, y la referencia de ΔE se renueva.
- **Pestaña "Condiciones iniciales"**: x, y, vx, vy de cada masa. Al soltar el slider (o
  pulsar Enter en el campo), la simulación vuelve a t = 0 desde esos valores y recalcula L.
  Con la simulación en pausa, mover un slider muestra una vista previa con las flechas de
  velocidad inicial.
- **Botones**: Iniciar/Pausar, Reiniciar (t = 0 con lo que hay en el panel; si estaba
  corriendo, sigue corriendo) y Valores por defecto. Los presets y los valores por defecto
  dejan la simulación en pausa en t = 0.
- **Presets**:
  - *Elástico sin fricción*: b = 0, e = 1; la energía total queda plana.
  - *Choques en las 4 paredes*: vel0 = [[8, 5], [−5, 7], [3, −6]]. Coincide con los valores
    por defecto, que son los del ejemplo del notebook.
  - *Respiración del triángulo*: masas y k iguales, triángulo equilátero, velocidades
    radiales, con g = 0 y b = 0 para ver el modo puro, ω0² = 3k/m.
- **Validación de entradas**: masas y h > 0; k, b y g ≥ 0; 0 ≤ e ≤ 1; posiciones dentro de la
  caja. Si un valor es inválido, el campo se pone rosado, el error aparece arriba del panel y
  la simulación sigue con el último valor válido.
- **Animación**: usa `root.after()` y no `FuncAnimation`. Cada cuadro mide el tiempo real
  transcurrido e integra `dt_real × velocidad / h` pasos de RK4 (≈ 33 pasos de 1 ms por cuadro
  a ×1). Así, pausar, cambiar la velocidad o reiniciar no exige recrear nada. Dibuja con
  blitting: el sistema se redibuja en cada cuadro y la energía cada 3 cuadros.
- **Rendimiento**: medido en este equipo, un paso tarda ~210 µs en Python puro. A ×1 la
  animación va a ~28 fps en tiempo real. A ×4 la CPU no alcanza: la GUI limita los pasos por
  cuadro y muestra la velocidad efectiva (≈ ×2.2 con h = 1 ms). La física no cambia, solo se
  ve más lenta que lo pedido.
