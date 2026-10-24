# Soluciones analíticas de ecuaciones diferenciales mediante programación genética

Implementación experimental de regresión simbólica con **NASA Bingo** para buscar soluciones analíticas de ecuaciones diferenciales ordinarias y en derivadas parciales. El proyecto replica los problemas de **Euler–Bernoulli** y **Poisson** estudiados por Oh *et al.* (2023).

El código principal se encuentra en un notebook de Jupyter (`.ipynb`). Este README explica cómo preparar el entorno, abrir el notebook y repetir los experimentos.

## Requisitos

- **Python 3.13** (versión utilizada durante el desarrollo).
- **Git** para clonar el repositorio.
- **uv** para crear el entorno e instalar dependencias. [Instrucciones de instalación de uv](https://docs.astral.sh/uv/getting-started/installation/).
- **Visual Studio Code**, con las extensiones **Python** y **Jupyter**, o JupyterLab.

Paquetes utilizados por el notebook:

| Paquete | Uso |
| --- | --- |
| `bingo-nasa==0.5.7` | Programación genética, árboles de expresiones e islas evolutivas |
| `numpy` | Operaciones numéricas y generación de puntos |
| `scipy` | Optimización local con Levenberg–Marquardt |
| `sympy` | Derivación simbólica de expresiones |
| `pandas` | Tablas y exportación de resultados |
| `matplotlib` | Gráficas |
| `ipykernel`, `jupyterlab` | Ejecución interactiva del notebook |

> **Nota:** `bingo-nasa` es el nombre del paquete que se instala; en Python se importa como `bingo`.

## 1. Clonar el repositorio

En una terminal, ejecutar:

```bash
git clone URL_DEL_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO
```

Sustituir `URL_DEL_REPOSITORIO` por la URL de GitHub y `NOMBRE_DEL_REPOSITORIO` por el nombre de la carpeta clonada.

## 2. Preparar el entorno de Python

### Opción A — El repositorio contiene `pyproject.toml` y `uv.lock` (recomendada)

Desde la raíz del proyecto:

```bash
uv sync
```

Esto instala las dependencias declaradas y utiliza las versiones del archivo de bloqueo. Si `ipykernel` y `jupyterlab` no están declarados como dependencias de desarrollo, agregarlos **una sola vez** con:

```bash
uv add --dev ipykernel jupyterlab
```

Después de configurar el proyecto, probar el entorno:

```bash
uv run python -c "import bingo, numpy, scipy, sympy, pandas, matplotlib, ipykernel; print('Dependencias OK')"
```

### Opción B — El repositorio no incluye configuración de uv

Crear un entorno e instalar las dependencias directamente:

```bash
uv venv --python 3.13
uv pip install "bingo-nasa==0.5.7" numpy scipy sympy pandas matplotlib ipykernel jupyterlab
```

No se necesita activar el entorno para instalar los paquetes con `uv pip` si la carpeta `.venv` está en la raíz del proyecto.

### Si `pyproject.toml` utiliza `vendor/bingo`

Algunas configuraciones instalan Bingo desde una copia local en `vendor/bingo`, en vez de PyPI. **Esa carpeta debe estar presente antes de ejecutar `uv sync`.** Si el repositorio la registra como submódulo de Git, descargarlo con:

```bash
git submodule update --init --recursive
```

Si no es un submódulo y la configuración requiere exactamente la versión 0.5.7, puede obtenerse manualmente con:

```bash
git clone --branch 0.5.7 https://github.com/nasa/bingo.git vendor/bingo
```

Solo ejecutar este último comando **si `vendor/bingo` no existe** y el proyecto efectivamente depende de esa ruta. Si la dependencia se resuelve desde PyPI, omitir todo este apartado.

## 3. Abrir y ejecutar el `.ipynb`

1. Abrir la carpeta del repositorio en **Visual Studio Code**.
2. Abrir el notebook principal, por ejemplo `EDO_GSPR.ipynb` (utilizar su nombre real si es distinto).
3. En la parte superior derecha, pulsar **Select Kernel / Seleccionar kernel**.
4. Elegir el intérprete Python ubicado dentro de la carpeta **`.venv`** del proyecto.
5. Ejecutar las celdas pertinentes en el orden indicado abajo.

**Importante:** el archivo `.html` que acompaña a una exportación de Jupyter permite **ver** el código y sus resultados, pero el archivo **ejecutable y editable** es el `.ipynb`.

### Orden recomendado de ejecución

El notebook incluye celdas exploratorias, pruebas anteriores y experimentos completos. Para reproducir las **30 corridas**, se necesitan las siguientes definiciones, en orden:

1. **Importaciones** de `numpy`, `pandas`, `random`, Bingo, `sympy`, `matplotlib` y demás módulos. Verificar también que se hayan ejecutado `import re` e `import time`.
2. **Euler–Bernoulli:** funciones `derivadas_sympy`, `_simbolicas` y `evalua`; después, la clase `EDO2`.
3. **Poisson:** funciones `_simbolicas_poisson`, `derivadas_poisson`, `evalua_nd` y `Make_Puntos`; después, la clase `EDO3`.
4. **Experimentos:** las celdas que definen `N_CORRIDAS`, `MAX_GENERACIONES`, `UMBRAL`, `experimento_poisson`, `experimento_euler` y `resumir_resultados`.
5. **Ejecución y exportación** de cada configuración, como se muestra a continuación.

> **Atención:** las celdas exploratorias antiguas no forman parte de la réplica de 30 corridas. En particular, una celda utiliza `EDO2(Xi, 1)`, pero la versión actual de `EDO2` recibe `n_datos` como entero. **No ejecutar esa celda** junto con el experimento final. Tampoco es necesario ejecutar los ejemplos comentados del comienzo.

## 4. Ejecutar los experimentos

### Poisson

Se admiten las dimensiones `d=1`, `d=2` y `d=3`. Ejemplo de una corrida experimental completa de 30 repeticiones independientes en una dimensión:

```python
resultados_1d = experimento_poisson(d=1)
resultados_1d.to_csv("resultados_poisson_1d.csv", index=False)
resumen_1d = resumir_resultados(resultados_1d)
```

Para otras dimensiones, ejecutar por separado:

```python
resultados_2d = experimento_poisson(d=2)
resultados_2d.to_csv("resultados_poisson_2d.csv", index=False)

resultados_3d = experimento_poisson(d=3)
resultados_3d.to_csv("resultados_poisson_3d.csv", index=False)
```

### Euler–Bernoulli

Se admiten `n_datos=2`, `3` u `11`, y dos conjuntos de operadores (`prueba=1` o `prueba=2`). Ejemplo:

```python
resultados_euler = experimento_euler(n_datos=2, prueba=1)
resultados_euler.to_csv("euler_2datos_test1.csv", index=False)
resumen_euler = resumir_resultados(resultados_euler)
```

Para reproducir otra configuración, cambiar los argumentos. Por ejemplo:

```python
resultados_euler_3 = experimento_euler(n_datos=3, prueba=2)
resultados_euler_3.to_csv("euler_3datos_test2.csv", index=False)
```

**Tiempo de ejecución:** cada experimento completo ejecuta 30 búsquedas evolutivas independientes. Las configuraciones de mayor complejidad pueden tardar considerablemente. Se recomienda ejecutar una configuración a la vez y guardar su CSV al terminar.

## 5. Parámetros experimentales

| Parámetro | Valor |
| --- | --- |
| Corridas independientes por configuración | `30` |
| Máximo de generaciones por corrida | `1000` |
| Umbral de convergencia del fitness | `1e-15` |
| Individuos por isla | `150` |
| Número de islas | `10` |
| Probabilidad de cruza | `0.5` |
| Probabilidad de mutación | `0.5` |
| Optimización local | Levenberg–Marquardt (`lm`) |
| Inicialización de constantes | `[-1, 1]` |
| Semillas por corrida | `1000` a `1029` |

El código usa los mismos puntos físicos dentro de una configuración experimental y modifica las semillas de la búsqueda genética para generar corridas independientes.

## 6. Interpretación de los resultados

Cada experimento devuelve un `pandas.DataFrame`, que puede exportarse a CSV.

| Columna | Significado |
| --- | --- |
| `corrida` | Número de repetición experimental |
| `seed` | Semilla aleatoria de la repetición |
| `fitness_final` | Error de entrenamiento: incumplimiento de la ecuación diferencial y sus condiciones |
| `mse_test` | Error cuadrático medio frente a la solución analítica en puntos de validación |
| `error_maximo` | Mayor error absoluto observado en la validación |
| `generaciones` | Generaciones ejecutadas antes de detener la búsqueda |
| `convergio` | Si el fitness alcanzó el umbral establecido |
| `tiempo_segundos` | Duración de la corrida |
| `expresion` | Mejor expresión simbólica encontrada |

La función `resumir_resultados(df)` calcula **media, mediana, desviación estándar** y **porcentaje de corridas que alcanzaron el umbral**.

**Importante:** `fitness_final` y `mse_test` no representan el mismo error. Un fitness próximo a cero indica que se cumplen las restricciones evaluadas; un MSE de prueba próximo a cero indica que la solución predice bien en el conjunto de validación. Alcanzar el umbral numérico **no demuestra por sí solo** equivalencia algebraica exacta.

Los CSV se guardan en el directorio de trabajo activo del kernel de Jupyter, normalmente la carpeta desde la que se abrió el proyecto.

## 7. Problemas frecuentes

- **`ModuleNotFoundError: No module named 'bingo'`:** comprobar que se seleccionó el kernel de `.venv` y que la instalación terminó correctamente.
- **`ModuleNotFoundError: No module named 'bingocpp'`:** revisar la instalación de `bingo-nasa`/`vendor/bingo` y la compatibilidad de Python. La disponibilidad del módulo compilado depende de la plataforma y del modo de instalación.
- **`NameError` de una función o clase:** ejecutar antes la celda que la define; las celdas de Jupyter comparten estado, pero no se inicializan automáticamente.
- **Error en `EDO2(Xi, 1)`:** es una celda exploratoria anterior. El experimento final construye correctamente `EDO2(n_datos=n_datos, lambdA=1)`.
- **No aparece el CSV:** comprobar el directorio activo con `import os; print(os.getcwd())` dentro del notebook.
- **La ejecución tarda mucho:** es esperable en experimentos con 30 repeticiones, 10 islas y derivación simbólica; comenzar por Poisson 1D para verificar la instalación.

## Referencia

H. Oh, R. Amici, G. Bomarito, S. Zhe, R. Kirby y J. Hochhalter, “Genetic programming based symbolic regression for analytical solutions to differential equations”, *arXiv*, arXiv:2302.03175, 2023. [https://doi.org/10.48550/arXiv.2302.03175](https://doi.org/10.48550/arXiv.2302.03175).

NASA Bingo: [https://github.com/nasa/bingo](https://github.com/nasa/bingo).
