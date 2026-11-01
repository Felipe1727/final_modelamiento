#!/usr/bin/env python3
"""
Sistema 5 — Tres masas unidas por resortes en triángulo, dentro de una caja con rebotes.
Proyecto final de Modelamiento Matemático (UPB).

Uso
    python sistema5.py             abre la GUI con la animación en vivo
    python sistema5.py --validar   corre las pruebas A–H y escribe PASA/FALLA

Todo está en un solo archivo, dividido en secciones:
    1. Modelo        RESORTES, Parametros, derivadas, energia        (notebook §2, §5.1, §5.2)
    2. Integrador    paso_rk4, paso_euler, rebotes, simular          (notebook §4, §5.3–§5.5)
    3. Simulador     estado actual + avanzar / reiniciar / actualizar (lo usa la animación)
    4. Analítica     soluciones exactas para validar                 (notebook §3)
    5. Validación    pruebas A–F del notebook + G y H (Simulador y entradas)
    6. GUI           Tkinter + Matplotlib embebido (FigureCanvasTkAgg)

Las ecuaciones, el esquema RK4 y la lógica de rebotes son los del notebook, sin cambios.
Único cambio de comportamiento: Parametros lanza ValueError (antes `assert`) si una masa
empieza fuera de la caja, para que la GUI pueda mostrar el error y para que `python -O`
no desactive la comprobación.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

try:                                  # la validación no necesita Tk; la GUI sí
    import tkinter as tk
    from tkinter import ttk
except ImportError:                   # pragma: no cover
    tk = ttk = None


# =====================================================================================
# 1. MODELO  (notebook §2 y §5.1–5.2)
# =====================================================================================

RESORTES: list[tuple[int, int]] = [(0, 1), (1, 2), (0, 2)]
"""Pares (i, j) que une cada resorte: resorte 1: m1–m2, resorte 2: m2–m3, resorte 3: m1–m3."""


@dataclass
class Parametros:
    """Los 20 parámetros del enunciado más las decisiones de diseño del modelo.

    Parámetros del enunciado (20):
        pos0 (3,2) posiciones iniciales [m]       -> 6
        vel0 (3,2) velocidades iniciales [m/s]    -> 6
        k    (3,)  k1, k2, k3 [N/m]               -> 3
        b          fricción lineal del aire [kg/s] -> 1   (fuerza -b·v, igual para las 3 masas)
        m    (3,)  m1, m2, m3 [kg]                -> 3
        g          gravedad [m/s²], en -y          -> 1

    Decisiones de diseño (el enunciado no las fija):
        ancho, alto  caja [0, W] x [0, H], con el piso en y = 0
        radio        las masas son discos de radio R (hace falta un tamaño para chocar)
        e            coeficiente de restitución (e = 1: choque elástico)
        L0           longitudes naturales; None -> distancias iniciales (el triángulo arranca relajado)
        rebotes      False = espacio libre, sin paredes (para las pruebas analíticas)
    """
    pos0: np.ndarray
    vel0: np.ndarray
    k: np.ndarray
    b: float
    m: np.ndarray
    g: float
    # --- decisiones de diseño (no están en el enunciado) ---
    ancho: float = 10.0
    alto: float = 6.0
    radio: float = 0.25
    e: float = 1.0
    L0: Optional[np.ndarray] = None
    rebotes: bool = True

    def __post_init__(self) -> None:
        self.pos0 = np.asarray(self.pos0, float).reshape(3, 2)
        self.vel0 = np.asarray(self.vel0, float).reshape(3, 2)
        self.k = np.asarray(self.k, float)
        self.m = np.asarray(self.m, float)
        if self.L0 is None:
            self.L0 = np.array([np.linalg.norm(self.pos0[i] - self.pos0[j]) for i, j in RESORTES])
        if self.rebotes:
            R = self.radio
            if not np.all((self.pos0[:, 0] >= R) & (self.pos0[:, 0] <= self.ancho - R) &
                          (self.pos0[:, 1] >= R) & (self.pos0[:, 1] <= self.alto - R)):
                raise ValueError('Las posiciones iniciales deben estar dentro de la caja '
                                 f'({R:g} <= x <= {self.ancho - R:g}, {R:g} <= y <= {self.alto - R:g} m)')

    def estado_inicial(self) -> np.ndarray:
        """S(0) = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]: las 12 condiciones iniciales."""
        return np.concatenate([self.pos0.ravel(), self.vel0.ravel()])

    def h_sugerido(self, puntos_por_periodo: int = 100) -> float:
        """Paso sugerido h = T_min / 100 (notebook §3.5 y §4.4).

        La oscilación más rápida tiene ω_max² = mayor autovalor de M⁻¹K. Para masas y k
        distintos se usa la cota conservadora (tipo Gershgorin) ω_max² ≲ 2.5·max_i(Σ_s k_s / m_i),
        sumando los resortes que tocan la masa i. Con T_min = 2π/ω_max, pedimos 100 puntos
        por período: así ω_max·h ≈ 0.06, muy lejos del límite de estabilidad de RK4 (≈ 2.83).
        """
        suma_k = np.zeros(3)
        for s, (i, j) in enumerate(RESORTES):
            suma_k[i] += self.k[s]; suma_k[j] += self.k[s]
        w_max = np.sqrt(2.5 * np.max(suma_k / self.m))
        return 2*np.pi / w_max / puntos_por_periodo if w_max > 0 else 1e-2


def derivadas(t: float, S: np.ndarray, p: Parametros) -> np.ndarray:
    """Lado derecho f(t, S) de las 12 EDOs de primer orden (notebook §2.3).

    dx_i/dt = vx_i,  dy_i/dt = vy_i,  dv_i/dt = F_i / m_i, con
        F_i = Σ_resortes −k_s (d_ij − L_s) û_ij   (Hooke a lo largo de la línea que une i y j)
              − b v_i                              (fricción lineal)
              − m_i g ŷ                            (gravedad)
    Cada resorte se calcula una sola vez y se suma con signos opuestos a sus dos masas
    (tercera ley de Newton). No depende de t (sistema autónomo), pero RK4 lo recibe igual.
    """
    r = S[:6].reshape(3, 2)          # posiciones
    v = S[6:].reshape(3, 2)          # velocidades
    F = np.zeros((3, 2))

    # Resortes: F_ij = -k (d - L) û_ij sobre i, y -F_ij sobre j
    for s, (i, j) in enumerate(RESORTES):
        delta = r[i] - r[j]
        dist = np.hypot(*delta)
        if dist > 1e-12:                         # evita dividir por 0 si coinciden
            f = -p.k[s] * (dist - p.L0[s]) * delta / dist
            F[i] += f
            F[j] -= f

    F -= p.b * v                     # fricción lineal
    F[:, 1] -= p.m * p.g             # gravedad en -y

    a = F / p.m[:, None]
    return np.concatenate([v.ravel(), a.ravel()])


def energia(S: np.ndarray, p: Parametros) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energías cinética, elástica y gravitatoria para cada fila de S (notebook §2.5).

    E_k = Σ ½ m_i |v_i|²,   E_e = Σ ½ k_s (d_s − L_s)²,   E_g = Σ m_i g y_i (referencia: piso).
    Se cumple dE/dt = −b Σ|v_i|² ≤ 0, así que con b = 0 y e = 1 la energía total se conserva.
    S debe ser 2D (N, 12); para un solo estado usar S[None, :].
    """
    r = S[:, :6].reshape(-1, 3, 2)
    v = S[:, 6:].reshape(-1, 3, 2)
    Ek = 0.5 * np.sum(p.m * np.sum(v**2, axis=2), axis=1)
    Ee = sum(0.5 * p.k[s] * (np.linalg.norm(r[:, i] - r[:, j], axis=1) - p.L0[s])**2
             for s, (i, j) in enumerate(RESORTES))
    Eg = np.sum(p.m * p.g * r[:, :, 1], axis=1)
    return Ek, Ee, Eg


# =====================================================================================
# 2. INTEGRADOR  (notebook §4 y §5.3–5.5)
# =====================================================================================

FuncionEDO = Callable[[float, np.ndarray, Parametros], np.ndarray]


def paso_rk4(f: FuncionEDO, t: float, S: np.ndarray, h: float, p: Parametros) -> np.ndarray:
    """Un paso de Runge–Kutta de orden 4 sobre el vector de estado completo (12 componentes).

    k1 = f(t, S)                  pendiente al inicio
    k2 = f(t + h/2, S + h/2·k1)   en el punto medio, usando k1
    k3 = f(t + h/2, S + h/2·k2)   en el punto medio, usando k2
    k4 = f(t + h, S + h·k3)       al final
    S_{n+1} = S_n + h/6·(k1 + 2k2 + 2k3 + k4)   (pesos de Simpson)
    Error local O(h⁵), error global O(h⁴). Cada k_i es un vector de 12 componentes:
    el método se aplica al sistema entero a la vez, lo que respeta el acoplamiento.
    """
    k1 = f(t,         S,             p)
    k2 = f(t + h/2,   S + h/2 * k1,  p)
    k3 = f(t + h/2,   S + h/2 * k2,  p)
    k4 = f(t + h,     S + h * k3,    p)
    return S + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def paso_euler(f: FuncionEDO, t: float, S: np.ndarray, h: float, p: Parametros) -> np.ndarray:
    """Un paso de Euler explícito, S_{n+1} = S_n + h·f(t_n, S_n). Solo para comparar.

    Error global O(h). En un oscilador sin fricción multiplica la amplitud por
    √(1 + ω²h²) > 1 en cada paso: la energía crece siempre (notebook §4.2).
    """
    return S + h * f(t, S, p)


V_REPOSO: float = 0.02
"""[m/s] Si la velocidad normal al tocar una pared es menor que esto, la masa queda apoyada."""


def aplicar_rebotes(S: np.ndarray, p: Parametros) -> np.ndarray:
    """Ley de choque para cada masa que está fuera de la caja (notebook §2.4).

    Pared izquierda/piso: v_n -> +e|v_n|;  pared derecha/techo: v_n -> −e|v_n|.
    El valor absoluto fija el signo hacia adentro y evita un doble rebote. La componente
    tangencial no cambia. Contacto en reposo: si |v_n| < V_REPOSO, la masa se coloca sobre
    la pared con v_n = 0 (evita la cascada infinita de rebotes, "paradoja de Zenón").
    Si la masa está fuera, también se refleja la posición; con eventos la penetración es
    ~1e-9 m, así que esta reflexión es despreciable. Modifica S en su lugar y lo devuelve.
    """
    r = S[:6].reshape(3, 2)          # vistas: modificar r y v modifica S
    v = S[6:].reshape(3, 2)
    R, e = p.radio, p.e
    limites = [(R, p.ancho - R), (R, p.alto - R)]     # eje 0 = x, eje 1 = y
    for i in range(3):
        for eje, (lo, hi) in enumerate(limites):
            for borde, adentro in [(lo, +1), (hi, -1)]:   # adentro = signo de la dirección hacia la caja
                if adentro * (borde - r[i, eje]) > 0:     # la masa está fuera de esta pared
                    if abs(v[i, eje]) < V_REPOSO:         # contacto en reposo
                        r[i, eje] = borde
                        v[i, eje] = 0.0
                    else:                                 # rebote
                        r[i, eje] = 2*borde - r[i, eje]
                        v[i, eje] = adentro * e * abs(v[i, eje])
    return S


def penetracion(S: np.ndarray, p: Parametros) -> float:
    """Función evento: la mayor distancia que una masa cruzó una pared moviéndose hacia afuera.

    Vale 0 si ninguna masa salió (o si sale más lento que V_REPOSO: eso es contacto en
    reposo, no un choque). La bisección de paso_con_eventos busca dónde deja de ser 0.
    """
    r = S[:6].reshape(3, 2); v = S[6:].reshape(3, 2); R = p.radio
    peor = 0.0
    for eje, (lo, hi) in enumerate([(R, p.ancho - R), (R, p.alto - R)]):
        for i in range(3):
            if r[i, eje] < lo and -v[i, eje] > V_REPOSO: peor = max(peor, lo - r[i, eje])
            if r[i, eje] > hi and  v[i, eje] > V_REPOSO: peor = max(peor, r[i, eje] - hi)
    return peor


def paso_con_eventos(t: float, S: np.ndarray, h: float, p: Parametros,
                     tol: float = 1e-9, max_eventos: int = 8) -> np.ndarray:
    """Un paso de tamaño h que localiza por bisección cada choque con las paredes (notebook §4.5).

    1. Se intenta el paso completo con RK4. Si ninguna masa salió de la caja, se acepta.
    2. Si alguna salió, se busca por bisección la fracción τ ∈ (0, h] en la que empieza la
       penetración, hasta un ancho de tol = 1e-9 s.
    3. Se avanza exactamente hasta τ, se aplica la ley de choque y se integra el resto h − τ,
       repitiendo por si hay otro choque (hasta max_eventos).
    Así el error de cada rebote baja de O(h) (rebote al final del paso) a ~tol.
    """
    restante = h
    for _ in range(max_eventos):
        S_try = paso_rk4(derivadas, t, S, restante, p)
        if penetracion(S_try, p) == 0:                 # sin choques: aceptar
            return aplicar_rebotes(S_try, p)           # (solo actúa si hay contacto en reposo)
        a, b = 0.0, restante                           # bisección del instante de choque
        while b - a > tol:
            mid = 0.5 * (a + b)
            if penetracion(paso_rk4(derivadas, t, S, mid, p), p) > 0:
                b = mid
            else:
                a = mid
        S = aplicar_rebotes(paso_rk4(derivadas, t, S, b, p), p)   # avanzar hasta el choque y rebotar
        t, restante = t + b, restante - b
        if restante <= tol:
            return S
    return aplicar_rebotes(paso_rk4(derivadas, t, S, restante, p), p)   # salvaguarda


def simular(p: Parametros, T: float, h: Optional[float] = None, metodo: str = 'rk4',
            eventos: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Integra de t = 0 a t = T en la malla t_n = n·h. Devuelve t (N+1,) y S (N+1, 12).

    metodo : 'rk4' o 'euler' (solo para comparar).
    eventos: True  -> rebotes con localización del instante de choque (recomendado)
             False -> rebote simple al final del paso (primer orden, para comparar)
    Si no se da h, se usa p.h_sugerido().
    """
    h = h or p.h_sugerido()
    paso = paso_rk4 if metodo == 'rk4' else paso_euler
    N = int(round(T / h))
    t = np.linspace(0, N*h, N + 1)
    S = np.empty((N + 1, 12))
    S[0] = p.estado_inicial()
    for n in range(N):
        if p.rebotes and eventos and metodo == 'rk4':
            S[n+1] = paso_con_eventos(t[n], S[n], h, p)
        else:
            S[n+1] = paso(derivadas, t[n], S[n], h, p)
            if p.rebotes:
                S[n+1] = aplicar_rebotes(S[n+1], p)
    return t, S


# =====================================================================================
# 3. SIMULADOR  (estado actual, para la animación)
# =====================================================================================

class Simulador:
    """Guarda el estado actual (t, S, p) y avanza paso a paso cuando la animación lo pide.

    `simular()` integra todo el intervalo de una vez y guarda la historia completa. La
    animación necesita otra cosa: avanzar unos pasos, dibujar, permitir cambios de
    parámetros y seguir. Cada paso es exactamente el de `simular()` (RK4 + eventos), así
    que la trayectoria es idéntica bit a bit (prueba G).
    """

    def __init__(self, p: Parametros, h: float = 1e-3) -> None:
        self.fijar_paso(h)
        self.reiniciar(p)

    def reiniciar(self, p: Optional[Parametros] = None) -> None:
        """Vuelve a t = 0 y S = S(0). Si se pasa un Parametros nuevo, se usan sus condiciones
        iniciales, y su L0 ya quedó recalculado como las distancias iniciales."""
        if p is not None:
            self.p = p
        self.t = 0.0
        self.S = self.p.estado_inicial()

    def fijar_paso(self, h: float) -> None:
        """Cambia el paso h de integración. No cambia la física, solo la discretización."""
        if not (np.isfinite(h) and h > 0):
            raise ValueError('h debe ser > 0')
        self.h = float(h)

    def avanzar(self, n_pasos: int = 1) -> np.ndarray:
        """Integra n_pasos pasos de tamaño h (RK4 + rebotes con eventos) y devuelve el estado S.

        Cada paso lee los parámetros de self.p en ese momento, así que un cambio hecho
        con actualizar_parametros() se aplica desde el paso siguiente.
        """
        for _ in range(n_pasos):
            if self.p.rebotes:
                self.S = paso_con_eventos(self.t, self.S, self.h, self.p)
            else:
                self.S = paso_rk4(derivadas, self.t, self.S, self.h, self.p)
            self.t += self.h
        return self.S

    def actualizar_parametros(self, k: Optional[Sequence[float]] = None, b: Optional[float] = None,
                              m: Optional[Sequence[float]] = None, g: Optional[float] = None,
                              e: Optional[float] = None) -> None:
        """Cambia k, b, m, g o e sin reiniciar. Se validan todos antes de aplicar alguno.

        Física del cambio en caliente: el estado S guarda posiciones y velocidades, así que
        se conservan x y v (no el momento m·v si cambia m). L0 no cambia: los resortes
        mantienen su longitud natural. Cambiar k, m o g cambia la energía potencial y la
        energía total "salta"; eso es esperado: el sistema ahora es otro.
        """
        nuevos: dict[str, object] = {}
        if k is not None:
            k = np.asarray(k, float).reshape(3)
            if not np.all(np.isfinite(k)) or np.any(k < 0):
                raise ValueError('k1, k2, k3 deben ser >= 0')
            nuevos['k'] = k
        if m is not None:
            m = np.asarray(m, float).reshape(3)
            if not np.all(np.isfinite(m)) or np.any(m <= 0):
                raise ValueError('m1, m2, m3 deben ser > 0')
            nuevos['m'] = m
        if b is not None:
            if not (np.isfinite(b) and b >= 0):
                raise ValueError('b debe ser >= 0')
            nuevos['b'] = float(b)
        if g is not None:
            if not (np.isfinite(g) and g >= 0):
                raise ValueError('g debe ser >= 0')
            nuevos['g'] = float(g)
        if e is not None:
            if not (np.isfinite(e) and 0 <= e <= 1):
                raise ValueError('e debe estar entre 0 y 1')
            nuevos['e'] = float(e)
        for nombre, valor in nuevos.items():
            setattr(self.p, nombre, valor)

    def energias(self) -> tuple[float, float, float]:
        """(E_cinética, E_elástica, E_gravitatoria) del estado actual, con los parámetros actuales."""
        Ek, Ee, Eg = energia(self.S[None, :], self.p)
        return float(Ek[0]), float(Ee[0]), float(Eg[0])


# =====================================================================================
# 4. SOLUCIONES ANALÍTICAS  (notebook §3, para validar)
# =====================================================================================

def oscilador_amortiguado(t, u0: float, v0: float, w0: float, gamma: float) -> np.ndarray:
    """Solución exacta de u'' + γu' + ω0²u = 0, u(0) = u0, u'(0) = v0 (notebook §3.2).

    Raíces λ = −γ/2 ± √(γ²/4 − ω0²) de la ecuación característica; u = C1 e^{λ1 t} + C2 e^{λ2 t}
    (sub- y sobreamortiguado, con raíces complejas) o (u0 + (v0 − λu0)t) e^{λt} (crítico).
    """
    t = np.asarray(t, float)
    disc = gamma**2 / 4 - w0**2
    if np.isclose(disc, 0.0):                       # crítico
        lam = -gamma / 2
        return (u0 + (v0 - lam*u0)*t) * np.exp(lam*t)
    raiz = np.sqrt(complex(disc))
    l1, l2 = -gamma/2 + raiz, -gamma/2 - raiz        # sub- o sobreamortiguado
    C1 = (v0 - l2*u0) / (l1 - l2)
    C2 = (l1*u0 - v0) / (l1 - l2)
    return np.real(C1*np.exp(l1*t) + C2*np.exp(l2*t))


def caso_respiracion(m: float = 1.0, k: float = 40.0, b: float = 0.3, g: float = 9.81,
                     Llado: float = 2.0, v_rad: float = 1.5, Vcm=(1.0, 2.0)):
    """Triángulo equilátero con velocidades radiales iguales + velocidad común del centro de masa.

    Por simetría el triángulo sigue siendo equilátero: la distancia al centroide ρ = ρ0 + u
    cumple u'' + γu' + (3k/m)u = 0 de forma exacta (notebook §3.4), y el centroide sigue el
    tiro con fricción lineal (§3.1). Devuelve (Parametros, función exacta t -> posiciones (N,3,2)).
    """
    rho0 = Llado / np.sqrt(3)
    angulos = np.deg2rad([90, 210, 330])
    e_hat = np.stack([np.cos(angulos), np.sin(angulos)], 1)        # direcciones centroide → vértice
    centro = np.array([5.0, 3.0])
    pos0 = centro + rho0 * e_hat
    vel0 = np.array(Vcm) + v_rad * e_hat
    p = Parametros(pos0=pos0, vel0=vel0, k=[k]*3, b=b, m=[m]*3, g=g, rebotes=False)

    def exacta(t):
        gam = b / m
        X = centro[0] + Vcm[0]/gam*(1 - np.exp(-gam*t))
        Y = centro[1] - g/gam*t + (Vcm[1] + g/gam)/gam*(1 - np.exp(-gam*t))
        u = oscilador_amortiguado(t, 0.0, v_rad, np.sqrt(3*k/m), gam)
        rho = rho0 + u
        return np.stack([np.stack([X + rho*e_hat[i, 0], Y + rho*e_hat[i, 1]], 1) for i in range(3)], 1)
    return p, exacta


# =====================================================================================
# 5. VALIDACIÓN  (notebook §5.6)
# =====================================================================================

Resultado = tuple[bool, list[str]]          # (pasa, líneas de detalle)


def prueba_A() -> Resultado:
    """Centro de masa vs. tiro con fricción lineal (§3.1). Masas iguales, sin paredes, resortes pre-estirados."""
    mA = 1.2
    pA = Parametros(pos0=[[3, 4], [5, 4.5], [4.5, 3]], vel0=[[2, 1], [-1, 0], [0.5, -2]],
                    k=[40, 30, 50], b=0.3, m=[mA]*3, g=9.81, rebotes=False)
    pA.L0 = pA.L0 * 0.8                        # el triángulo vibra con fuerza
    tA, SA = simular(pA, T=3.0, h=1e-3)
    R_num = np.stack([SA[:, 0:6:2].mean(1), SA[:, 1:6:2].mean(1)], 1)
    R0, V0 = pA.pos0.mean(0), pA.vel0.mean(0)
    gam = pA.b / mA
    X_an = R0[0] + V0[0]/gam*(1 - np.exp(-gam*tA))
    Y_an = R0[1] - pA.g/gam*tA + (V0[1] + pA.g/gam)/gam*(1 - np.exp(-gam*tA))
    err = np.hypot(R_num[:, 0] - X_an, R_num[:, 1] - Y_an).max()
    return err < 1e-10, [f'error máx. del centro de masa = {err:.2e} m   (tol 1e-10 m)']


def prueba_B() -> Resultado:
    """Un resorte entre dos masas iguales (§3.3): u = d − L oscila con ω0² = 2k/m."""
    mB, kB, bB = 1.0, 40.0, 0.4
    pB = Parametros(pos0=[[3, 4], [5, 4], [8, 2]], vel0=[[-1.5, 2.0], [1.5, 2.0], [0, 0]],
                    k=[kB, 0, 0], b=bB, m=[mB, mB, 1.0], g=9.81, rebotes=False)
    tB, SB = simular(pB, T=5.0, h=1e-3)
    u_num = np.hypot(SB[:, 2] - SB[:, 0], SB[:, 3] - SB[:, 1]) - pB.L0[0]
    u_an = oscilador_amortiguado(tB, 0.0, 3.0, np.sqrt(2*kB/mB), bB/mB)
    err = np.abs(u_num - u_an).max()
    return err < 1e-8, [f'error máx. de la deformación u = {err:.2e} m   (tol 1e-8 m)']


def prueba_C() -> Resultado:
    """Modo de respiración del triángulo equilátero (§3.4): las 12 EDOs completas vs. solución exacta."""
    pC, exactaC = caso_respiracion()
    tC, SC = simular(pC, T=4.0, h=1e-3)
    err = np.abs(SC[:, :6].reshape(-1, 3, 2) - exactaC(tC)).max()
    return err < 1e-8, [f'error máx. en las 6 coordenadas = {err:.2e} m   (tol 1e-8 m)']


def prueba_D() -> Resultado:
    """Orden de convergencia en el caso C: error ≈ C·h^p; la pendiente log–log da p."""
    pC, exactaC = caso_respiracion()
    hs = np.array([0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625])
    err_rk4, err_eu = [], []
    for hh in hs:
        for metodo, lista in [('rk4', err_rk4), ('euler', err_eu)]:
            tt_, SS = simular(pC, T=2.0, h=hh, metodo=metodo)
            lista.append(np.abs(SS[:, :6].reshape(-1, 3, 2) - exactaC(tt_)).max())
    err_rk4, err_eu = np.array(err_rk4), np.array(err_eu)
    p_rk4 = np.polyfit(np.log(hs[-3:]), np.log(err_rk4[-3:]), 1)[0]
    p_eu = np.polyfit(np.log(hs[-3:]), np.log(err_eu[-3:]), 1)[0]
    lineas = [' h          error RK4   razón   error Euler']
    for i, hh in enumerate(hs):
        razon = err_rk4[i-1]/err_rk4[i] if i else float('nan')
        lineas.append(f' {hh:<10g} {err_rk4[i]:.3e}   {razon:5.1f}   {err_eu[i]:.3e}')
    lineas.append(f'pendiente RK4 = {p_rk4:.2f} (esperado 4 ± 0.2)   ·   '
                  f'pendiente Euler = {p_eu:.2f} (esperado 1 ± 0.2)')
    return abs(p_rk4 - 4) < 0.2 and abs(p_eu - 1) < 0.2, lineas


def prueba_E() -> Resultado:
    """Conservación de energía con b = 0, e = 1 y rebotes: rebote simple vs. localización del evento."""
    pE = Parametros(pos0=[[3, 4], [5, 4.5], [4.5, 3]], vel0=[[8, 5], [-5, 7], [3, -6]],
                    k=[40, 30, 50], b=0.0, m=[1.0, 1.5, 0.8], g=9.81, e=1.0)
    lineas = [' h        |ΔE/E0| máx. en 10 s:  rebote simple   con eventos']
    pasa = True
    for hh in [2e-3, 1e-3]:
        fila = []
        for ev in [False, True]:
            _, SE = simular(pE, T=10.0, h=hh, eventos=ev)
            E = sum(energia(SE, pE))
            fila.append(np.abs((E - E[0]) / abs(E[0])).max())
        pasa &= fila[1] < 1e-6 and fila[1] < fila[0]
        lineas.append(f' {hh:<8g}                        {fila[0]:.2e}        {fila[1]:.2e}')
    lineas.append('criterio: con eventos |ΔE/E0| < 1e-6 y mejor que el rebote simple')
    return pasa, lineas


def prueba_F() -> Resultado:
    """Contacto en reposo (e = 0.7, b = 0.5): la energía nunca aumenta y el sistema queda quieto."""
    pF = Parametros(pos0=[[3, 4], [5, 4.5], [4.5, 3]], vel0=[[2, 1], [-1, 0], [0.5, -2]],
                    k=[40, 30, 50], b=0.5, m=[1.0, 1.5, 0.8], g=9.81, e=0.7)
    _, SF = simular(pF, T=30.0, h=1e-3)
    EF = sum(energia(SF, pF))
    nunca_sube = bool(np.all(np.diff(EF) <= 1e-9 * abs(EF[0])))
    v_fin = np.abs(SF[-1, 6:]).max()
    r_fin = SF[-1, :6].reshape(3, 2)
    dentro = bool(np.all((r_fin >= pF.radio - 1e-9) &
                         (r_fin <= np.array([pF.ancho, pF.alto]) - pF.radio + 1e-9)))
    return nunca_sube and v_fin < 1e-2 and dentro, [
        f'E nunca aumenta: {nunca_sube}   ·   |v| máx. final = {v_fin:.1e} m/s (tol 1e-2)   ·   '
        f'masas dentro de la caja: {dentro}']


def prueba_G() -> Resultado:
    """(Extra) El Simulador que usa la GUI da lo mismo que simular(), y los cambios en vivo funcionan.

    El tramo con b = 0 dura 2 s (t = 1 → 3 s). En t ≈ 3.75 s, m1 y m2 se cruzan a 2 cm (no hay
    choques entre masas). Con L > 0 la fuerza −k(d − L)û cambia de dirección de golpe en d = 0,
    y RK4 con h = 1 ms no resuelve ese cruce. Es un límite del modelo, no del Simulador, así que
    la prueba se queda antes de ese instante y muestra la distancia mínima alcanzada.
    """
    base = dict(pos0=[[3, 4], [5, 4.5], [4.5, 3]], vel0=[[8, 5], [-5, 7], [3, -6]],
                k=[40, 30, 50], b=0.1, m=[1.0, 1.5, 0.8], g=9.81, e=1.0)
    _, S_ref = simular(Parametros(**base), T=2.0, h=1e-3)
    sim = Simulador(Parametros(**base), h=1e-3)
    sim.avanzar(2000)
    identico = bool(np.array_equal(sim.S, S_ref[-1]))
    # Cambio en vivo: tras 1 s con fricción, poner b = 0 y e = 1 -> desde ahí E se conserva.
    sim = Simulador(Parametros(**base), h=1e-3)
    sim.avanzar(1000)
    E_antes = sum(sim.energias())
    sim.actualizar_parametros(b=0.0, e=1.0)
    E1 = sum(sim.energias())
    d_min = np.inf
    for _ in range(2000):
        r = sim.avanzar(1)[:6].reshape(3, 2)
        d_min = min(d_min, min(np.linalg.norm(r[i] - r[j]) for i, j in RESORTES))
    deriva = abs(sum(sim.energias()) - E1) / abs(E1)
    disipo = bool(E_antes < sum(energia(S_ref[:1], Parametros(**base)))[0])
    return identico and deriva < 1e-6 and disipo, [
        f'Simulador.avanzar(2000) ≡ simular(T=2): {identico} (igualdad exacta)',
        f'con b = 0.1 la energía bajó en el primer segundo: {disipo}',
        f'tras poner b = 0, e = 1 en vivo: |ΔE/E| = {deriva:.1e} en 2 s (tol 1e-6; '
        f'distancia mínima entre masas = {d_min:.2f} m)']


def prueba_H() -> Resultado:
    """(Extra) Las entradas inválidas se rechazan con ValueError (la GUI las muestra sin caerse)."""
    base = dict(pos0=[[3, 4], [5, 4.5], [4.5, 3]], vel0=[[0, 0]]*3, k=[40, 30, 50],
                b=0.1, m=[1.0, 1.5, 0.8], g=9.81)
    casos: list[tuple[str, Callable[[], object]]] = [
        ('posición fuera de la caja', lambda: Parametros(**{**base, 'pos0': [[3, 4], [12, 4.5], [4.5, 3]]})),
        ('masa = 0', lambda: Simulador(Parametros(**base)).actualizar_parametros(m=[1, 0, 1])),
        ('k < 0', lambda: Simulador(Parametros(**base)).actualizar_parametros(k=[40, -1, 50])),
        ('b < 0', lambda: Simulador(Parametros(**base)).actualizar_parametros(b=-0.1)),
        ('g < 0', lambda: Simulador(Parametros(**base)).actualizar_parametros(g=-9.81)),
        ('e > 1', lambda: Simulador(Parametros(**base)).actualizar_parametros(e=1.5)),
        ('h = 0', lambda: Simulador(Parametros(**base), h=0.0)),
    ]
    rechazados = []
    for nombre, f in casos:
        try:
            f()
        except ValueError:
            rechazados.append(nombre)
    return len(rechazados) == len(casos), [f'rechazados {len(rechazados)}/{len(casos)}: ' + ', '.join(rechazados)]


PRUEBAS: list[tuple[str, str, Callable[[], Resultado]]] = [
    ('A', 'Centro de masa vs. analítica (§3.1)', prueba_A),
    ('B', 'Un resorte, ω0² = 2k/m (§3.3)', prueba_B),
    ('C', 'Modo de respiración, ω0² = 3k/m (§3.4)', prueba_C),
    ('D', 'Orden de convergencia RK4 ≈ 4, Euler ≈ 1', prueba_D),
    ('E', 'Energía con b = 0, e = 1 y rebotes', prueba_E),
    ('F', 'Contacto en reposo (e = 0.7, b = 0.5)', prueba_F),
    ('G', 'Simulador ≡ simular + cambios en vivo', prueba_G),
    ('H', 'Validación de entradas', prueba_H),
]


def validar() -> bool:
    """Corre todas las pruebas, escribe PASA/FALLA y devuelve True si todas pasan."""
    print('Validación de sistema5.py: discretización del notebook (RK4 + rebotes con eventos)\n')
    n_pasa = 0
    for letra, titulo, prueba in PRUEBAS:
        t0 = time.perf_counter()
        try:
            pasa, lineas = prueba()
        except Exception as exc:                      # una prueba rota cuenta como FALLA
            pasa, lineas = False, [f'excepción: {exc!r}']
        n_pasa += pasa
        print(f'[{"PASA " if pasa else "FALLA"}] {letra}  {titulo}   ({time.perf_counter() - t0:.1f} s)')
        for linea in lineas:
            print('         ' + linea)
    print(f'\nResultado: {n_pasa}/{len(PRUEBAS)} pruebas pasan.')
    return n_pasa == len(PRUEBAS)


# =====================================================================================
# 6. GUI  (Tkinter + Matplotlib embebido)
# =====================================================================================
#
# ¿Por qué root.after() y no FuncAnimation?
#   FuncAnimation tiene su propio temporizador y, para pausar, cambiar la velocidad o
#   reiniciar, hay que detenerla o recrearla. Con after() el bucle es nuestro: cada cuadro
#   mide el tiempo real transcurrido, integra los pasos de RK4 que corresponden a ese tiempo
#   (× velocidad) y dibuja con *blitting*: el fondo estático (caja, ejes, leyendas) se
#   guarda una vez y en cada cuadro solo se redibujan las masas, los resortes, las estelas
#   y las curvas de energía. Los cambios de parámetros llegan como eventos de Tk entre dos
#   cuadros, así que se aplican desde el siguiente paso sin condiciones de carrera.

COLORES = ['#2a78d6', '#eb6834', '#1baf7a']          # m1, m2, m3 (fijos, como en el notebook)
TINTA, TINTA_2, TINTA_3 = '#222222', '#555555', '#8a8a8a'
C_ESTIRADO, C_COMPRIMIDO = '#c0392b', '#7d3c98'      # color del resorte: rojo = estirado, morado = comprimido
C_ENERGIA = {'cinética': '#eda100', 'elástica': '#e87ba4', 'gravitatoria': '#008300', 'total': TINTA}
C_BIEN, C_AVISO, C_ERROR = '#1a7f37', '#b35900', '#b42318'

FPS = 30                       # cuadros por segundo pedidos
PRESUPUESTO = 0.022            # [s] tiempo máx. de cálculo por cuadro (deja ~11 ms para dibujar)
VENTANA_E = 10.0               # [s simulados] ancho de la ventana del panel de energía
ESTELA_S = 1.0                 # [s simulados] largo de la estela
ESCALA_FLECHA = 6.0            # flechas de velocidad inicial: 1 m de flecha = 6 m/s
CUADROS_ENERGIA = 3            # el panel de energía se redibuja cada 3 cuadros (10 Hz)

DEFECTO = dict(                # valores por defecto = ejemplo del notebook (celda "Simulación de ejemplo")
    pos0=[[3.0, 4.0], [5.0, 4.5], [4.5, 3.0]],
    vel0=[[8.0, 5.0], [-5.0, 7.0], [3.0, -6.0]],
    k=[40.0, 30.0, 50.0], b=0.1, m=[1.0, 1.5, 0.8], g=9.81, e=1.0, h=1e-3)


def preset_elastico() -> dict:
    """b = 0 y e = 1 con choques en las cuatro paredes: la energía total debe quedar plana (prueba E)."""
    return {**DEFECTO, 'b': 0.0, 'e': 1.0}


def preset_cuatro_paredes() -> dict:
    """Velocidades grandes que llevan a las masas a chocar con las cuatro paredes (valores del notebook)."""
    return {**DEFECTO, 'vel0': [[8.0, 5.0], [-5.0, 7.0], [3.0, -6.0]]}


def preset_respiracion() -> dict:
    """Masas y k iguales, triángulo equilátero y velocidades radiales (§3.4).

    Se usa g = 0 y b = 0: así el centroide no se mueve, ninguna masa toca las paredes y se ve
    el modo puro con ω0² = 3k/m. Subir b en vivo muestra el decaimiento e^{−γt/2}.
    """
    m, k, lado, v_rad = 1.0, 40.0, 2.0, 3.0
    ang = np.deg2rad([90, 210, 330])
    e_hat = np.stack([np.cos(ang), np.sin(ang)], 1)
    pos0 = np.array([5.0, 3.0]) + lado / np.sqrt(3) * e_hat
    return dict(pos0=pos0.tolist(), vel0=(v_rad * e_hat).tolist(), k=[k]*3, b=0.0,
                m=[m]*3, g=0.0, e=1.0, h=1e-3)


def puntos_resorte(a: np.ndarray, b: np.ndarray, L0: float, n: int = 12,
                   amp0: float = 0.12, extremo: float = 0.15) -> np.ndarray:
    """Polilínea en zigzag entre a y b, solo para dibujar (no entra en la física).

    El "alambre" de cada diente tiene largo fijo, calculado con la longitud natural L0. Al
    comprimir el resorte el zigzag se ensancha y al estirarlo se aplana, como en un
    resorte real.
    """
    d = b - a
    dist = float(np.hypot(*d))
    if dist < 1e-9:
        return np.array([a, b])
    u = d / dist
    nrm = np.array([-u[1], u[0]])
    frac = 1 - 2*extremo
    alambre = np.hypot(frac*max(L0, 1e-9)/n, 2*amp0)
    amp = min(0.5*np.sqrt(max(alambre**2 - (frac*dist/n)**2, 0.0)), 0.35)
    s = np.concatenate([[0, extremo], extremo + frac*(np.arange(n) + 0.5)/n, [1 - extremo, 1]])
    lateral = np.concatenate([[0, 0], amp*np.where(np.arange(n) % 2 == 0, 1, -1), [0, 0]])
    return a + np.outer(s, d) + np.outer(lateral, nrm)


def estilo_resorte(dist: float, L0: float) -> tuple[tuple[float, float, float], float]:
    """Color y grosor según la deformación relativa ε = (d − L0)/L0.

    Gris = longitud natural, rojo = estirado, morado = comprimido. El color satura en |ε| = 25 %.
    """
    from matplotlib.colors import to_rgb
    eps = (dist - L0) / max(L0, 1e-9)
    s = float(np.clip(eps / 0.25, -1, 1))
    base = np.array(to_rgb(TINTA_2))
    destino = np.array(to_rgb(C_ESTIRADO if s > 0 else C_COMPRIMIDO))
    return tuple(base + abs(s)*(destino - base)), 1.3 + 1.4*abs(s)


class Control:
    """Una fila de la GUI: etiqueta con unidades, slider (ttk.Scale) y campo numérico (ttk.Entry).

    El slider y el campo están sincronizados. `valor` siempre guarda el último valor válido
    en unidades SI. Si se escribe un valor inválido, el campo se marca en rojo, el error se
    muestra en la barra de estado y `valor` no cambia, así que el programa sigue funcionando.
    El campo acepta valores fuera del rango del slider si son físicamente válidos.
    """

    def __init__(self, padre, fila: int, texto: str, desde: float, hasta: float, valor: float, *,
                 paso: float, fmt: str, validar: Callable[[float], Optional[str]],
                 al_cambiar: Callable[['Control'], None], al_soltar: Optional[Callable[['Control'], None]] = None,
                 al_error: Callable[[str], None], escala: float = 1.0, nombre: Optional[str] = None) -> None:
        self.texto, self.paso, self.fmt, self.escala = texto, paso, fmt, escala
        self.nombre = nombre or texto            # cómo se llama el parámetro en los mensajes de error
        self.validar, self.al_cambiar, self.al_soltar, self.al_error = validar, al_cambiar, al_soltar, al_error
        self._silencio = False
        self._texto_mostrado = ''
        ttk.Label(padre, text=texto).grid(row=fila, column=0, sticky='w', padx=(6, 6))
        self.slider = ttk.Scale(padre, from_=desde, to=hasta, orient='horizontal', length=160,
                                command=self._al_mover)
        self.slider.grid(row=fila, column=1, sticky='ew', pady=1)
        self.var = tk.StringVar()
        self.campo = ttk.Entry(padre, textvariable=self.var, width=8, justify='right')
        self.campo.grid(row=fila, column=2, padx=(6, 6), pady=1)
        for evento in ('<Return>', '<KP_Enter>', '<FocusOut>'):
            self.campo.bind(evento, self._al_escribir)
        self.slider.bind('<ButtonRelease-1>', self._al_soltar_slider)
        self.slider.bind('<KeyRelease>', self._al_soltar_slider)
        self.fijar(valor)

    def fijar(self, valor: float) -> None:
        """Pone un valor desde el programa (presets, valores por defecto) sin disparar callbacks."""
        self.valor = float(valor)
        self._silencio = True
        self.slider.set(self.valor * self.escala)
        self._silencio = False
        self._mostrar()

    def _mostrar(self) -> None:
        self._texto_mostrado = self.fmt.format(self.valor * self.escala)
        self.var.set(self._texto_mostrado)
        self.campo.state(['!invalid'])

    def _al_mover(self, texto: str) -> None:
        if self._silencio:
            return
        v = round(round(float(texto) / self.paso) * self.paso, 10)    # ajusta a la resolución
        self.valor = v / self.escala
        self._mostrar()
        self.al_cambiar(self)

    def _al_soltar_slider(self, _evento=None) -> None:
        if self.al_soltar:
            self.al_soltar(self)

    def _al_escribir(self, _evento=None) -> None:
        texto = self.var.get().strip().replace(',', '.')
        if texto == self._texto_mostrado:          # nada cambió (p. ej. solo se movió el foco)
            return
        try:
            v = float(texto) / self.escala
            if not np.isfinite(v):
                raise ValueError
        except ValueError:
            self.campo.state(['invalid'])
            self.al_error(f'{self.nombre}: «{texto}» no es un número')
            return
        msg = self.validar(v)
        if msg:
            self.campo.state(['invalid'])
            self.al_error(f'{self.nombre}: {msg}')
            return
        self.fijar(v)
        self.al_cambiar(self)
        if self.al_soltar:
            self.al_soltar(self)


def _mayor_que_0(v: float) -> Optional[str]:
    return None if v > 0 else 'debe ser > 0'


def _mayor_o_igual_0(v: float) -> Optional[str]:
    return None if v >= 0 else 'debe ser >= 0'


def _entre(lo: float, hi: float) -> Callable[[float], Optional[str]]:
    return lambda v: None if lo <= v <= hi else f'debe estar entre {lo:g} y {hi:g}'


class App:
    """Ventana principal: panel de parámetros a la izquierda y figura animada a la derecha."""

    def __init__(self, root: 'tk.Tk') -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.collections import LineCollection

        self.root = root
        root.title('Sistema 5 — tres masas, tres resortes, caja con rebotes')
        root.report_callback_exception = self._excepcion_tk     # ningún error tumba la GUI
        estilo = ttk.Style(root)
        if 'clam' in estilo.theme_names():
            estilo.theme_use('clam')
        estilo.map('TEntry', fieldbackground=[('invalid', '#fde2e2')], foreground=[('invalid', C_ERROR)])
        estilo.configure('Nota.TLabel', foreground=TINTA_2)

        self.p_ref = Parametros(**{k: v for k, v in DEFECTO.items() if k != 'h'})   # caja y radio
        W, H, R = self.p_ref.ancho, self.p_ref.alto, self.p_ref.radio

        # ---------------- panel de controles ----------------
        panel = ttk.Frame(root, padding=(8, 8, 4, 8))
        panel.grid(row=0, column=0, sticky='ns')
        ttk.Label(panel, text='Sistema 5 · 3 masas, 3 resortes, caja con rebotes',
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
        # Errores y avisos arriba, siempre visibles (aunque la pantalla sea baja).
        self.lbl_estado = ttk.Label(panel, text='', wraplength=350, justify='left')
        self.lbl_estado.pack(anchor='w', fill='x', pady=(2, 4))

        pestanas = self.pestanas = ttk.Notebook(panel)
        pestanas.pack(fill='x')
        fisica = ttk.Frame(pestanas, padding=4)
        inicial = ttk.Frame(pestanas, padding=4)
        pestanas.add(fisica, text='Física (en vivo)')
        pestanas.add(inicial, text='Condiciones iniciales')

        comun = dict(al_error=self._mostrar_error)
        vivo = dict(al_cambiar=self._cambio_fisico, **comun)
        sub = '₁₂₃'

        def grupo(padre, titulo):
            f = ttk.LabelFrame(padre, text=titulo, padding=(2, 2))
            f.pack(fill='x', pady=3)
            return f

        f = grupo(fisica, 'Resortes')
        self.c_k = [Control(f, s, f'k{s+1} [N/m]', 0, 200, DEFECTO['k'][s], paso=1, fmt='{:.1f}',
                            validar=_mayor_o_igual_0, **vivo) for s in range(3)]
        f = grupo(fisica, 'Masas')
        self.c_m = [Control(f, i, f'm{i+1} [kg]', 0.1, 5, DEFECTO['m'][i], paso=0.05, fmt='{:.2f}',
                            validar=_mayor_que_0, **vivo) for i in range(3)]
        f = grupo(fisica, 'Entorno')
        self.c_b = Control(f, 0, 'b [kg/s]', 0, 3, DEFECTO['b'], paso=0.01, fmt='{:.2f}',
                           validar=_mayor_o_igual_0, **vivo)
        self.c_g = Control(f, 1, 'g [m/s²]', 0, 25, DEFECTO['g'], paso=0.01, fmt='{:.2f}',
                           validar=_mayor_o_igual_0, **vivo)
        self.c_e = Control(f, 2, 'e (restitución)', 0, 1, DEFECTO['e'], paso=0.01, fmt='{:.2f}',
                           validar=_entre(0, 1), **vivo)
        f = grupo(fisica, 'Integración numérica')
        self.c_h = Control(f, 0, 'h [ms]', 0.1, 10, DEFECTO['h'], paso=0.05, fmt='{:.2f}', escala=1e3,
                           validar=_mayor_que_0, al_cambiar=self._cambio_paso, **comun)
        self.lbl_h = ttk.Label(f, text='', justify='left', wraplength=330)
        self.lbl_h.grid(row=1, column=0, columnspan=3, sticky='w', padx=6, pady=(2, 2))
        ttk.Label(fisica, style='Nota.TLabel', wraplength=340, justify='left',
                  text='Se aplican desde el siguiente paso, sin reiniciar. Las longitudes '
                       'naturales L no cambian.').pack(anchor='w', pady=(2, 0))

        ci = dict(al_cambiar=self._cambio_inicial, al_soltar=self._soltar_inicial, **comun)
        self.c_pos, self.c_vel = [], []
        for i in range(3):
            f = grupo(inicial, f'm{i+1}')
            tk.Frame(f, width=10, height=10, bg=COLORES[i]).grid(row=0, column=3, padx=(0, 4))
            self.c_pos.append([
                Control(f, 0, 'x [m]', R, W - R, DEFECTO['pos0'][i][0], paso=0.05, fmt='{:.2f}',
                        validar=_entre(R, W - R), nombre=f'x{i+1} [m]', **ci),
                Control(f, 1, 'y [m]', R, H - R, DEFECTO['pos0'][i][1], paso=0.05, fmt='{:.2f}',
                        validar=_entre(R, H - R), nombre=f'y{i+1} [m]', **ci)])
            self.c_vel.append([
                Control(f, 2, 'vx [m/s]', -15, 15, DEFECTO['vel0'][i][0], paso=0.1, fmt='{:.2f}',
                        validar=lambda v: None, nombre=f'vx{i+1} [m/s]', **ci),
                Control(f, 3, 'vy [m/s]', -15, 15, DEFECTO['vel0'][i][1], paso=0.1, fmt='{:.2f}',
                        validar=lambda v: None, nombre=f'vy{i+1} [m/s]', **ci)])
        self.lbl_L0 = ttk.Label(inicial, text='', style='Nota.TLabel')
        self.lbl_L0.pack(anchor='w')
        ttk.Label(inicial, style='Nota.TLabel', wraplength=340, justify='left',
                  text='Al soltar el slider (o Enter) vuelve a t = 0 y recalcula L = distancias '
                       'iniciales. En pausa, mover un slider muestra la vista previa.'
                  ).pack(anchor='w', pady=(2, 0))

        botones = ttk.Frame(panel)
        botones.pack(fill='x', pady=(6, 2))
        self.btn_play = ttk.Button(botones, text='Iniciar', command=self.alternar, width=9)
        self.btn_play.pack(side='left')
        ttk.Button(botones, text='Reiniciar', command=self.reiniciar).pack(side='left', padx=4)
        ttk.Button(botones, text='Valores por defecto',
                   command=lambda: self.cargar(DEFECTO)).pack(side='left')

        f = ttk.Frame(panel)
        f.pack(fill='x', pady=2)
        self.velocidad = 1.0
        ttk.Label(f, text='Velocidad').grid(row=0, column=0, padx=(0, 4))
        ttk.Scale(f, from_=-2, to=2, value=0, orient='horizontal', length=120,
                  command=self._cambio_velocidad).grid(row=0, column=1)
        self.lbl_vel = ttk.Label(f, text='×1.00', width=6)
        self.lbl_vel.grid(row=0, column=2, padx=4)
        self.lbl_vel_ef = ttk.Label(f, text='', style='Nota.TLabel')
        self.lbl_vel_ef.grid(row=0, column=3, sticky='w')

        f = ttk.LabelFrame(panel, text='Presets para la demo', padding=2)
        f.pack(fill='x', pady=2)
        f.columnconfigure((0, 1), weight=1)
        for (fila, col, ancho), texto, preset in [
                ((0, 0, 2), 'Elástico sin fricción (b = 0, e = 1)', preset_elastico),
                ((1, 0, 1), 'Choques en las 4 paredes', preset_cuatro_paredes),
                ((1, 1, 1), 'Respiración del triángulo', preset_respiracion)]:
            ttk.Button(f, text=texto, command=lambda pr=preset: self.cargar(pr())).grid(
                row=fila, column=col, columnspan=ancho, sticky='ew', padx=1, pady=1)

        # ---------------- figura ----------------
        self.fig = Figure(figsize=(8.2, 7.4), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1.35], hspace=0.32,
                                   left=0.08, right=0.98, top=0.95, bottom=0.07)
        ax = self.ax_sis = self.fig.add_subplot(gs[0])
        axE = self.ax_E = self.fig.add_subplot(gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew')
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        from matplotlib.patches import Circle, Rectangle
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, lw=1.6, color=TINTA))
        ax.set_xlim(-0.3, W + 0.3); ax.set_ylim(-0.3, H + 0.3); ax.set_aspect('equal')
        ax.set_xlabel('x, y en m   ·   resorte: gris = longitud natural, rojo = estirado, '
                      'morado = comprimido   ·   flechas = velocidad inicial', color=TINTA_2, fontsize=8)
        ax.set_ylabel('y [m]', color=TINTA_2)
        ax.tick_params(colors=TINTA_3, labelsize=8)
        ax.grid(alpha=0.15)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)
        ax.annotate('', xy=(W - 0.5, H - 1.4), xytext=(W - 0.5, H - 0.5),
                    arrowprops=dict(arrowstyle='->', color=TINTA_2, lw=1.4))
        ax.text(W - 0.35, H - 1.0, 'g', color=TINTA_2, fontsize=11, style='italic')

        self.a_estelas = [ax.add_collection(LineCollection([], linewidths=1.6, zorder=1)) for _ in range(3)]
        self.a_resortes = [ax.plot([], [], lw=1.3, color=TINTA_2, zorder=2,
                                   solid_joinstyle='round')[0] for _ in range(3)]
        self.a_masas = [ax.add_patch(Circle((0, 0), R, facecolor=COLORES[i], edgecolor='white',
                                            lw=1.5, zorder=3)) for i in range(3)]
        self.a_etiquetas = [ax.text(0, 0, f'm{sub[i]}', ha='center', va='center', fontsize=9,
                                    color=TINTA, zorder=5) for i in range(3)]
        self.a_flechas = ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), color=COLORES,
                                   angles='xy', scale_units='xy', scale=ESCALA_FLECHA, width=0.005,
                                   zorder=4)
        self.a_tiempo = ax.text(0.0, 1.02, '', transform=ax.transAxes, fontsize=12, color=TINTA,
                                family='monospace')
        self.a_modo = ax.text(1.0, 1.02, '', transform=ax.transAxes, ha='right', fontsize=9, color=TINTA_2)

        self.a_E = {nombre: axE.plot([], [], color=c, lw=2.6 if nombre == 'total' else 1.8,
                                     label=nombre)[0] for nombre, c in C_ENERGIA.items()}
        self.a_marcas = axE.plot([], [], color=TINTA_3, lw=1, ls='--')[0]
        self.a_lectura = axE.text(1.0, 0.96, '', transform=axE.transAxes, ha='right', va='top',
                                  fontsize=8.5, color=TINTA, family='monospace')
        axE.legend(loc='upper left', ncol=4, fontsize=8, frameon=False, handlelength=1.6,
                   borderaxespad=0.2, columnspacing=1.0)
        axE.set_xlabel('t [s]', color=TINTA_2); axE.set_ylabel('E [J]', color=TINTA_2)
        axE.tick_params(colors=TINTA_3, labelsize=8)
        axE.grid(alpha=0.25)
        for lado in ('top', 'right'):
            axE.spines[lado].set_visible(False)
        axE.set_title('Energía (marcas punteadas = cambio de parámetros en vivo)', fontsize=9,
                      color=TINTA_2, loc='left')

        # Dos regiones de blitting: el sistema se redibuja en cada cuadro y el panel de energía
        # cada CUADROS_ENERGIA cuadros (dibujar texto es lo más caro en Agg). El texto de modo
        # (corriendo / en pausa) no está animado: va en el fondo y solo se redibuja al cambiar.
        self.animados_sis = [*self.a_estelas, *self.a_resortes, *self.a_masas, *self.a_etiquetas,
                             self.a_flechas, self.a_tiempo]
        self.animados_E = [*self.a_E.values(), self.a_marcas, self.a_lectura]
        for a in self.animados_sis + self.animados_E:
            a.set_animated(True)
        self._fondos = None
        self._texto_modo = ''
        self.canvas.mpl_connect('draw_event', self._al_dibujar)

        # ---------------- estado ----------------
        self.corriendo = False
        self.vista_previa = False
        self._sucio = True
        self._costo_paso = 2e-4          # [s] estimación del tiempo de CPU de un paso (se ajusta sola)
        self._vel_efectiva = 1.0
        self._cuadro = 0
        self.sim = Simulador(self._parametros_panel(), h=self.c_h.valor)
        self._reiniciar_historia()
        self._actualizar_indicador_h()

        root.bind('<space>', self._tecla_espacio)
        root.protocol('WM_DELETE_WINDOW', self._cerrar)
        self.canvas.draw()
        self._id_after = root.after(50, self._cuadro_animacion)

    # ======================= lectura del panel =======================

    def _valores_panel(self) -> dict:
        return dict(pos0=[[c.valor for c in fila] for fila in self.c_pos],
                    vel0=[[c.valor for c in fila] for fila in self.c_vel],
                    k=[c.valor for c in self.c_k], b=self.c_b.valor, m=[c.valor for c in self.c_m],
                    g=self.c_g.valor, e=self.c_e.valor, h=self.c_h.valor)

    def _parametros_panel(self) -> Parametros:
        """Construye un Parametros nuevo (L0 = distancias iniciales) con los valores del panel."""
        v = self._valores_panel()
        v.pop('h')
        return Parametros(**v)

    # ======================= acciones =======================

    def alternar(self) -> None:
        """Botón Iniciar/Pausar."""
        self.corriendo = not self.corriendo
        self.vista_previa = False
        self._t_reloj = time.perf_counter()
        self._deuda = 0.0
        self.btn_play.config(text='Pausar' if self.corriendo else 'Iniciar')
        self._sucio = True

    def reiniciar(self) -> None:
        """Vuelve a t = 0 con las condiciones iniciales y los parámetros del panel.

        Recalcula L0 = distancias iniciales. Si estaba corriendo, sigue corriendo.
        """
        try:
            p = self._parametros_panel()
            self.sim.fijar_paso(self.c_h.valor)
        except ValueError as exc:
            self._mostrar_error(str(exc))
            return
        self.sim.reiniciar(p)
        self.vista_previa = False
        self._reiniciar_historia()
        self._limpiar_estado()
        self._actualizar_indicador_h()
        self._t_reloj = time.perf_counter()
        self._deuda = 0.0

    def cargar(self, valores: dict) -> None:
        """Carga un preset o los valores por defecto, reinicia y deja la simulación en pausa en t = 0."""
        for i in range(3):
            for eje in range(2):
                self.c_pos[i][eje].fijar(valores['pos0'][i][eje])
                self.c_vel[i][eje].fijar(valores['vel0'][i][eje])
            self.c_k[i].fijar(valores['k'][i])
            self.c_m[i].fijar(valores['m'][i])
        self.c_b.fijar(valores['b']); self.c_g.fijar(valores['g'])
        self.c_e.fijar(valores['e']); self.c_h.fijar(valores['h'])
        if self.corriendo:
            self.alternar()
        self.reiniciar()

    # ======================= callbacks de los controles =======================

    def _cambio_fisico(self, _control=None) -> None:
        """k, b, m, g, e: se aplican en vivo desde el siguiente paso (Simulador.actualizar_parametros)."""
        try:
            self.sim.actualizar_parametros(k=[c.valor for c in self.c_k], b=self.c_b.valor,
                                           m=[c.valor for c in self.c_m], g=self.c_g.valor,
                                           e=self.c_e.valor)
        except ValueError as exc:
            self._mostrar_error(str(exc))
            return
        self._limpiar_estado()
        self._actualizar_indicador_h()
        self._marcar_cambio()
        self._sucio = True

    def _cambio_paso(self, _control=None) -> None:
        """h: cambia la discretización, no la física; tampoco reinicia."""
        try:
            self.sim.fijar_paso(self.c_h.valor)
        except ValueError as exc:
            self._mostrar_error(str(exc))
            return
        self._limpiar_estado()
        self._actualizar_indicador_h()

    def _cambio_inicial(self, _control=None) -> None:
        """Mientras se arrastra un slider de condición inicial: vista previa si está en pausa."""
        if not self.corriendo:
            self.vista_previa = True
            pos = np.array([[c.valor for c in fila] for fila in self.c_pos])
            self._mostrar_L0(self._distancias(pos))
            self._sucio = True

    def _soltar_inicial(self, _control=None) -> None:
        """Al soltar el slider (o Enter): reinicia desde las nuevas condiciones iniciales."""
        self.reiniciar()

    def _cambio_velocidad(self, texto: str) -> None:
        self.velocidad = 2.0 ** float(texto)
        self.lbl_vel.config(text=f'×{self.velocidad:.2f}')

    def _tecla_espacio(self, evento) -> None:
        if not isinstance(evento.widget, (ttk.Entry, ttk.Button, tk.Entry)):
            self.alternar()

    # ======================= bucle de animación =======================

    def _cuadro_animacion(self) -> None:
        """Un cuadro: integra los pasos que corresponden al tiempo real transcurrido y redibuja."""
        inicio = time.perf_counter()
        try:
            redibujar_todo = False
            if self.corriendo:
                redibujar_todo = self._integrar(inicio)
            if self._sucio or self.corriendo:
                energia_toca = self._sucio or self._cuadro % CUADROS_ENERGIA == 0
                self._actualizar_artistas(energia_toca)
                redibujar_todo |= self._actualizar_modo()
                if redibujar_todo:
                    self.canvas.draw()          # cambiaron los ejes o el modo: redibujo completo (+ fondo)
                else:
                    self._blit(energia_toca)
                self._sucio = False
            self._cuadro += 1
            if self._cuadro % 10 == 0:
                self.lbl_vel_ef.config(text=f'efectiva ×{self._vel_efectiva:.2f}'
                                       if self.corriendo else '')
        except Exception as exc:                # nunca dejar morir el bucle
            self._detener_por_error(f'Error en la animación: {exc}')
            traceback.print_exc()
        espera = 1000 / FPS - (time.perf_counter() - inicio) * 1000
        self._id_after = self.root.after(max(1, int(espera)), self._cuadro_animacion)

    def _integrar(self, ahora: float) -> bool:
        """Avanza la simulación. Devuelve True si hay que redibujar los ejes de energía.

        Pasos por cuadro: el tiempo simulado debe avanzar dt_real × velocidad, es decir,
        n = dt_real·velocidad / h pasos (≈ 33 pasos de 1 ms por cuadro a 30 fps y ×1). Si la
        CPU no alcanza (h muy chico o velocidad alta), se limita n para no congelar la
        ventana; la simulación sigue siendo exacta, solo se ve más lenta que la velocidad
        pedida (se muestra la "velocidad efectiva").
        """
        dt_real = min(ahora - self._t_reloj, 0.1)
        self._t_reloj = ahora
        h = self.sim.h
        self._deuda += dt_real * self.velocidad
        n = int(self._deuda / h)
        n_max = max(1, int(PRESUPUESTO / self._costo_paso))
        if n > n_max:
            n, self._deuda = n_max, 0.0
        else:
            self._deuda -= n * h
        if n == 0:
            return False

        t0 = time.perf_counter()
        trozos = min(4, n)                     # 4 puntos de estela por cuadro como máximo
        for c in range(trozos):
            self.sim.avanzar(n // trozos + (1 if c < n % trozos else 0))
            if not np.all(np.isfinite(self.sim.S)) or np.abs(self.sim.S).max() > 1e6:
                self._detener_por_error('La simulación divergió (valores no finitos o enormes). '
                                        'Reduce h (mira el h sugerido) o las constantes k, y reinicia.')
                return False
            self.estela.append((self.sim.t, self.sim.S[:6].reshape(3, 2).copy()))
        costo = (time.perf_counter() - t0) / n
        self._costo_paso = 0.8 * self._costo_paso + 0.2 * costo
        if dt_real > 0:
            self._vel_efectiva = 0.9 * self._vel_efectiva + 0.1 * (n * h / dt_real)
        while self.estela and self.estela[0][0] < self.sim.t - ESTELA_S:
            self.estela.popleft()
        return self._muestrear_energia()

    # ======================= energía =======================

    def _reiniciar_historia(self) -> None:
        self.estela: deque = deque()
        self.hist_t: list[float] = []
        self.hist_E: list[tuple[float, float, float, float]] = []
        self.marcas: list[float] = []
        self._E_ref: Optional[float] = None
        Ek, Ee, Eg = self.sim.energias()
        self._E_top = 1.3 * max(Ek + Ee + Eg, 1e-3)
        self.ax_E.set_xlim(0, VENTANA_E)
        self.ax_E.set_ylim(0, self._E_top)
        self._muestrear_energia()
        self._mostrar_L0(self.sim.p.L0)
        self.canvas.draw_idle()
        self._sucio = True

    def _muestrear_energia(self) -> bool:
        """Guarda (E_k, E_e, E_g, E_total) en t actual. Devuelve True si cambiaron los ejes."""
        Ek, Ee, Eg = self.sim.energias()
        Et = Ek + Ee + Eg
        if self._E_ref is None:              # referencia para ΔE: tras reiniciar o cambiar parámetros
            self._E_ref = Et
        t = self.sim.t
        self.hist_t.append(t)
        self.hist_E.append((Ek, Ee, Eg, Et))
        cambio = False
        x0, x1 = self.ax_E.get_xlim()
        if t > x1:                           # desplazar la ventana: se ven los últimos VENTANA_E/2 s
            x0 = t - VENTANA_E / 2
            self.ax_E.set_xlim(x0, x0 + VENTANA_E)
            while self.hist_t and self.hist_t[0] < x0:
                self.hist_t.pop(0); self.hist_E.pop(0)
            self._E_top = 1.3 * max(max(fila) for fila in self.hist_E)
            cambio = True
        if max(Ek, Ee, Eg, Et) > self._E_top / 1.1:
            self._E_top = 1.3 * max(Ek, Ee, Eg, Et)
            cambio = True
        if cambio:
            self.ax_E.set_ylim(0, max(self._E_top, 1e-3))
        return cambio

    def _marcar_cambio(self) -> None:
        """Marca vertical en el panel de energía al cambiar parámetros; la referencia de ΔE se renueva."""
        t = self.sim.t
        if t > 0 and (not self.marcas or t - self.marcas[-1] > 0.05):
            self.marcas.append(t)
        self._E_ref = None

    # ======================= dibujo =======================

    def _regiones(self):
        """Dos franjas horizontales de la figura: arriba el sistema, abajo el panel de energía."""
        from matplotlib.transforms import Bbox
        ancho, alto = self.fig.bbox.width, self.fig.bbox.height
        corte = self.ax_E.bbox.y1 + 2
        return Bbox.from_extents(0, corte, ancho, alto), Bbox.from_extents(0, 0, ancho, corte)

    def _al_dibujar(self, _evento=None) -> None:
        """Tras cada redibujo completo (inicio, cambio de tamaño, cambio de ejes): guardar el fondo."""
        reg_sis, reg_E = self._regiones()
        self._fondos = (reg_sis, self.canvas.copy_from_bbox(reg_sis),
                        reg_E, self.canvas.copy_from_bbox(reg_E))
        for a in self.animados_sis + self.animados_E:
            self.fig.draw_artist(a)

    def _blit(self, energia_toca: bool = True) -> None:
        """Restaura el fondo guardado y dibuja encima solo los objetos animados de cada región."""
        if self._fondos is None:
            self.canvas.draw()
            return
        reg_sis, fondo_sis, reg_E, fondo_E = self._fondos
        self.canvas.restore_region(fondo_sis)
        for a in self.animados_sis:
            self.fig.draw_artist(a)
        self.canvas.blit(reg_sis)
        if energia_toca:
            self.canvas.restore_region(fondo_E)
            for a in self.animados_E:
                self.fig.draw_artist(a)
            self.canvas.blit(reg_E)

    def _actualizar_modo(self) -> bool:
        """Texto 'corriendo / en pausa / vista previa'. Devuelve True si cambió (hay que redibujar el fondo)."""
        if self.vista_previa:
            texto = 'vista previa · suelta el slider para aplicar'
        elif self.corriendo:
            texto = '▶ corriendo'
        else:
            texto = 'en pausa'
        if texto == self._texto_modo:
            return False
        self._texto_modo = texto
        self.a_modo.set_text(texto)
        return True

    @staticmethod
    def _distancias(pos: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in RESORTES])

    def _actualizar_artistas(self, energia_toca: bool = True) -> None:
        """Pasa el estado actual (o la vista previa) a los objetos de Matplotlib; no dibuja."""
        R = self.sim.p.radio
        if self.vista_previa:                # posiciones del panel; resortes relajados (L0 = d)
            pos = np.array([[c.valor for c in fila] for fila in self.c_pos])
            vel = np.array([[c.valor for c in fila] for fila in self.c_vel])
            L0 = self._distancias(pos)
            flechas = True
        else:
            pos = self.sim.S[:6].reshape(3, 2)
            vel = self.sim.S[6:].reshape(3, 2)
            L0 = self.sim.p.L0
            flechas = self.sim.t == 0.0

        for s, (i, j) in enumerate(RESORTES):
            pts = puntos_resorte(pos[i], pos[j], L0[s])
            color, grosor = estilo_resorte(float(np.hypot(*(pos[i] - pos[j]))), L0[s])
            self.a_resortes[s].set_data(pts[:, 0], pts[:, 1])
            self.a_resortes[s].set_color(color)
            self.a_resortes[s].set_linewidth(grosor)
        centroide = pos.mean(axis=0)
        for i in range(3):
            self.a_masas[i].center = tuple(pos[i])
            afuera = pos[i] - centroide          # etiqueta hacia afuera del triángulo: no la cruzan resortes
            n = np.hypot(*afuera)
            afuera = afuera / n if n > 1e-9 else np.array([0.0, 1.0])
            self.a_etiquetas[i].set_position(tuple(pos[i] + (R + 0.2) * afuera))

        self.a_flechas.set_offsets(pos)
        self.a_flechas.set_UVC(vel[:, 0], vel[:, 1])
        self.a_flechas.set_visible(flechas)

        from matplotlib.colors import to_rgba
        puntos = [] if self.vista_previa else [q for _, q in self.estela]
        for i in range(3):
            if len(puntos) >= 2:
                trayecto = np.array([q[i] for q in puntos] + [pos[i]])
                segs = np.stack([trayecto[:-1], trayecto[1:]], axis=1)
                colores = np.tile(to_rgba(COLORES[i]), (len(segs), 1))
                colores[:, 3] = np.linspace(0.0, 0.6, len(segs))     # se desvanece hacia atrás
                self.a_estelas[i].set_segments(segs)
                self.a_estelas[i].set_color(colores)
            else:
                self.a_estelas[i].set_segments([])

        self.a_tiempo.set_text('t = 0.000 s' if self.vista_previa else f't = {self.sim.t:7.3f} s')

        if not energia_toca:
            return
        if self.hist_t:
            t = np.array(self.hist_t)
            E = np.array(self.hist_E)
            for col, nombre in enumerate(C_ENERGIA):
                self.a_E[nombre].set_data(t, E[:, col])
            Et = E[-1, 3]
            rel = (Et - self._E_ref) / abs(self._E_ref) if self._E_ref else 0.0
            p = self.sim.p
            nota = 'se conserva' if p.b == 0 and p.e == 1 else 'disipa'
            self.a_lectura.set_text(f'E = {Et:.3f} J   ΔE/E_ref = {rel:+.1e}   ({nota})')
        ytop = self.ax_E.get_ylim()[1]
        xs = [v for t in self.marcas for v in (t, t, np.nan)]
        ys = [v for _ in self.marcas for v in (0, ytop, np.nan)]
        self.a_marcas.set_data(xs, ys)

    # ======================= mensajes =======================

    def _mostrar_L0(self, L0: np.ndarray) -> None:
        self.lbl_L0.config(text='L1, L2, L3 = ' + ', '.join(f'{x:.3f}' for x in L0) + ' m')

    def _actualizar_indicador_h(self) -> None:
        """h sugerido = T_min/100 y avisos: h > sugerido, inestabilidad de RK4, micro-rebotes."""
        p, h = self.sim.p, self.sim.h
        h_sug = p.h_sugerido()
        hay_resortes = bool(np.any(p.k > 0))
        w_max = 2 * np.pi / (100 * h_sug) if hay_resortes else 0.0
        lineas = [f'h sugerido (T_min/100) = {h_sug*1e3:.2f} ms' + ('' if hay_resortes else ' (sin resortes)')]
        if w_max * h > 2.83:
            color = C_ERROR
            lineas.append(f'Error: w_max·h = {w_max*h:.2f} > 2.83, RK4 puede volverse inestable')
        elif h > h_sug:
            color = C_AVISO
            lineas.append(f'Aviso: h = {h*1e3:.2f} ms > h sugerido, menos precisión')
        else:
            color = C_BIEN
            lineas.append(f'OK: h = {h*1e3:.2f} ms <= h sugerido')
        if p.g > 0 and h > V_REPOSO / p.g:
            if color == C_BIEN:
                color = C_AVISO
            lineas.append(f'Aviso: g·h > V_REPOSO = {V_REPOSO} m/s, el contacto en reposo puede '
                          'mostrar micro-rebotes')
        self.lbl_h.config(text='\n'.join(lineas), foreground=color)

    def _mostrar_error(self, msg: str) -> None:
        self.lbl_estado.config(text='Error: ' + msg, foreground=C_ERROR)

    def _limpiar_estado(self) -> None:
        """Borra el error anterior; avisa si hay masas superpuestas (no se modelan choques entre masas)."""
        pos = self.sim.p.pos0
        sup = [f'm{i+1}-m{j+1}' for i, j in RESORTES
               if np.linalg.norm(pos[i] - pos[j]) < 2 * self.sim.p.radio]
        if sup:
            self.lbl_estado.config(text='Aviso: masas superpuestas al inicio (' + ', '.join(sup) +
                                   '): el modelo no incluye choques entre masas.', foreground=C_AVISO)
        else:
            self.lbl_estado.config(text='')

    def _detener_por_error(self, msg: str) -> None:
        if self.corriendo:
            self.alternar()
        self._mostrar_error(msg)

    def _excepcion_tk(self, tipo, valor, tb) -> None:
        """Cualquier excepción en un callback de Tk se muestra en la GUI en vez de cerrar el programa."""
        traceback.print_exception(tipo, valor, tb)
        self._detener_por_error(f'{tipo.__name__}: {valor}')

    def _cerrar(self) -> None:
        try:
            self.root.after_cancel(self._id_after)
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()


def ejecutar_gui() -> None:
    """Crea la ventana y entra al bucle de eventos de Tk."""
    if tk is None:
        sys.exit('Tkinter no está disponible en esta instalación de Python.')
    root = tk.Tk()
    App(root)
    root.mainloop()


# =====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sistema 5: tres masas, tres resortes, caja con rebotes.')
    parser.add_argument('--validar', action='store_true', help='corre las pruebas A–H y sale')
    args = parser.parse_args()
    if args.validar:
        sys.exit(0 if validar() else 1)
    ejecutar_gui()
