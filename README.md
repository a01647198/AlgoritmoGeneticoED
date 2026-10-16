# PR-GPSR para EDOs — proyecto inicial para laptop

Este proyecto implementa una versión didáctica del enfoque del artículo:

> H. Oh et al., *Genetic Programming Based Symbolic Regression for Analytical Solutions to Differential Equations*.

La idea central es evolucionar **expresiones matemáticas** en lugar de vectores numéricos.

## Problema de ejemplo

Se intenta descubrir una expresión `u(x)` que satisfaga:

```text
u''(x) = -π² sin(πx),   0 < x < 1
u(0) = 0
u(1) = 0
```

La solución exacta `sin(πx)` NO se utiliza en la función de fitness. Solamente se utiliza después para comprobar si el método realmente encontró la solución.

## 1. Requisitos

Recomendado: Python 3.11 o 3.12.

### Con uv

```bash
uv venv
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Después:

```bash
uv pip install numpy scipy sympy matplotlib
```

### Con pip

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Después:

```bash
python -m pip install --upgrade pip
pip install numpy scipy sympy matplotlib
```

## 2. Ejecutar

```bash
python pr_gpsr_ode.py
```

También puedes cambiar parámetros:

```bash
python pr_gpsr_ode.py --seed 67 --population 150 --generations 100 --collocation 21 --max-nodes 12
```

Se crea una carpeta `results/` con:

- `solution.png`
- `convergence.png`

## 3. Qué hace el algoritmo

Cada individuo es un árbol de expresión como:

```text
sin(c0 * x)
```

o:

```text
c0*x + c1*sin(c2*x)
```

Las estructuras evolucionan mediante:

- selección,
- crossover de subárboles,
- mutación de subárboles.

Cada estructura contiene constantes libres `c0, c1, ...`.
Esas constantes se optimizan localmente con `scipy.optimize.least_squares`,
siguiendo la idea del artículo.

La función de fitness contiene:

1. Residual de la EDO en puntos interiores:

```text
r_phys(xj) = u''(xj) + π² sin(πxj)
```

2. Residuales de frontera:

```text
r_bc1 = u(0)
r_bc2 = u(1)
```

3. MSE del vector completo de residuales.

## 4. Cómo cambiar a otra EDO

Busca dentro de `evaluate_individual()` esta línea:

```python
r_phys = u2 + (math.pi**2) * np.sin(math.pi * x_phys)
```

y sustitúyela por el residual de tu ecuación.

Ejemplo:

```text
u'' + 4u = 0
```

requiere calcular también `u(x_phys)` y usar:

```python
r_phys = u2 + 4.0*u
```

Para condiciones iniciales que incluyan derivadas, por ejemplo:

```text
u(0)=0
u'(0)=2
```

calcula además:

```python
d1_expr = sp.diff(expr, X, 1)
```

haz `lambdify` de `d1_expr`, evalúalo en 0 y añade ese residual al vector de fitness.

## 5. Para acercarte más al artículo

Este ejemplo simplifica varias cosas. El artículo usa:

- Bingo de NASA.
- Deterministic crowding.
- Varias islas evolutivas.
- Diferenciación automática.
- Optimización local de coeficientes.
- Repetición de experimentos debido a la estocasticidad.

La versión de este proyecto usa selección por torneo y SymPy porque es más fácil de entender, depurar y ejecutar en una laptop.

## 6. Recomendación experimental

Primero haz funcionar problemas sencillos de 1 variable.
Después agrega, uno por uno:

1. deterministic crowding;
2. múltiples islas;
3. más operadores;
4. ecuaciones de orden superior;
5. sistemas de EDOs;
6. PDEs.

No agregues muchos operadores desde el inicio: el espacio de búsqueda crece muy rápido.
