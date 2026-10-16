"""
PR-GPSR didáctico para descubrir una solución analítica de una EDO.

Ejemplo:
    u''(x) = -pi^2 sin(pi x),   x in (0, 1)
    u(0) = 0
    u(1) = 0

La solución conocida es u(x)=sin(pi x), PERO NO se utiliza para entrenar.
Solo se usa al final para validación.

Inspirado en:
Oh et al., "Genetic Programming Based Symbolic Regression for
Analytical Solutions to Differential Equations" (2023).

Este script implementa una versión pequeña para laptop:
- Programación genética con árboles de expresiones.
- Operadores +, -, *, sin, cos.
- Constantes libres.
- Fitness regularizado por la física: residual de la EDO + condiciones de frontera.
- Optimización local de constantes con scipy.optimize.least_squares.
- Diferenciación simbólica con SymPy.
- Selección por torneo (más simple que deterministic crowding del artículo).

No pretende ser una reproducción bit-a-bit de Bingo/NASA.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.optimize import least_squares


# ============================================================
# 1. Representación de los individuos: árboles de expresiones
# ============================================================

@dataclass
class Node:
    op: str
    children: tuple["Node", ...] = ()
    value: Optional[float] = None


def tree_size(t: Node) -> int:
    return 1 + sum(tree_size(c) for c in t.children)


def tree_depth(t: Node) -> int:
    if not t.children:
        return 1
    return 1 + max(tree_depth(c) for c in t.children)


def random_tree(max_depth: int = 3, p_terminal: float = 0.30) -> Node:
    """
    Crea un árbol aleatorio.

    Terminales:
      x  -> variable independiente
      c  -> constante real ajustable

    Operadores:
      +, -, *, sin, cos
    """
    if max_depth <= 0 or random.random() < p_terminal:
        if random.random() < 0.55:
            return Node("x")
        return Node("c", value=random.uniform(-1.0, 1.0))

    # Aproximadamente 35% de los nodos no terminales serán unarios.
    if random.random() < 0.35:
        op = random.choice(["sin", "cos"])
        return Node(op, (random_tree(max_depth - 1, p_terminal),))

    op = random.choice(["add", "sub", "mul"])
    return Node(
        op,
        (
            random_tree(max_depth - 1, p_terminal),
            random_tree(max_depth - 1, p_terminal),
        ),
    )


# ============================================================
# 2. Utilidades para crossover y mutación de subárboles
# ============================================================

def all_paths(t: Node, prefix=()):
    """Devuelve todas las rutas posibles hacia subárboles."""
    result = [prefix]
    for i, child in enumerate(t.children):
        result.extend(all_paths(child, prefix + (i,)))
    return result


def get_subtree(t: Node, path):
    node = t
    for i in path:
        node = node.children[i]
    return node


def replace_subtree(t: Node, path, new_subtree: Node) -> Node:
    """Devuelve una copia del árbol con el subárbol indicado reemplazado."""
    if not path:
        return copy.deepcopy(new_subtree)

    t = copy.deepcopy(t)

    def rec(node: Node, p):
        i = p[0]
        children = list(node.children)
        if len(p) == 1:
            children[i] = copy.deepcopy(new_subtree)
        else:
            children[i] = rec(children[i], p[1:])
        node.children = tuple(children)
        return node

    return rec(t, path)


def crossover(
    a: Node,
    b: Node,
    max_nodes: int,
    max_depth_allowed: int,
):
    """Crossover de subárboles."""
    pa = random.choice(all_paths(a))
    pb = random.choice(all_paths(b))

    child_a = replace_subtree(a, pa, get_subtree(b, pb))
    child_b = replace_subtree(b, pb, get_subtree(a, pa))

    if tree_size(child_a) > max_nodes or tree_depth(child_a) > max_depth_allowed:
        child_a = copy.deepcopy(a)

    if tree_size(child_b) > max_nodes or tree_depth(child_b) > max_depth_allowed:
        child_b = copy.deepcopy(b)

    return child_a, child_b


def mutate(
    a: Node,
    max_nodes: int,
    max_depth_allowed: int,
) -> Node:
    """Reemplaza un subárbol por otro generado aleatoriamente."""
    p = random.choice(all_paths(a))
    new_subtree = random_tree(max_depth=random.randint(0, 2), p_terminal=0.35)
    mutant = replace_subtree(a, p, new_subtree)

    if tree_size(mutant) > max_nodes or tree_depth(mutant) > max_depth_allowed:
        return copy.deepcopy(a)

    return mutant


# ============================================================
# 3. Conversión a SymPy
# ============================================================

X = sp.Symbol("x")


def to_sympy(t: Node, const_nodes):
    """
    Convierte el árbol a una expresión SymPy.

    Cada nodo constante se transforma temporalmente en c0, c1, c2, ...
    para permitir la optimización local de sus valores.
    """
    if t.op == "x":
        return X

    if t.op == "c":
        symbol = sp.Symbol(f"c{len(const_nodes)}")
        const_nodes.append((symbol, t))
        return symbol

    args = [to_sympy(c, const_nodes) for c in t.children]

    if t.op == "add":
        return args[0] + args[1]
    if t.op == "sub":
        return args[0] - args[1]
    if t.op == "mul":
        return args[0] * args[1]
    if t.op == "sin":
        return sp.sin(args[0])
    if t.op == "cos":
        return sp.cos(args[0])

    raise ValueError(f"Operador desconocido: {t.op}")


def snap_known_constants(expr):
    """
    Intenta reconocer constantes especiales después de la evolución.
    Por ejemplo, 3.14159265358979 -> pi y 1.5707963267949 -> pi/2.

    Esto es SOLO postprocesamiento simbólico; no participa en el fitness.
    """
    replacements = {}
    for value in expr.atoms(sp.Float):
        try:
            replacements[value] = sp.nsimplify(
                value,
                constants=[sp.pi, sp.E],
                tolerance=1e-10,
                full=True,
            )
        except Exception:
            pass
    return sp.simplify(expr.xreplace(replacements))


def expression_with_values(t: Node, snap_constants: bool = False):
    const_nodes = []
    expr = to_sympy(t, const_nodes)
    substitutions = {symbol: node.value for symbol, node in const_nodes}
    expr = sp.simplify(expr.subs(substitutions))
    return snap_known_constants(expr) if snap_constants else expr


# ============================================================
# 4. Problema diferencial y fitness physics-regularized
# ============================================================

def make_collocation_points(seed: int, n_points: int):
    """
    Puntos interiores aleatorios.

    Usar puntos aleatorios ayuda a reducir soluciones espurias que solamente
    "engañan" una malla regular.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.02, 0.98, size=n_points))


def _as_vector(value, n):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return np.broadcast_to(arr, (n,)).astype(float)


def evaluate_individual(
    individual: Node,
    x_phys: np.ndarray,
    bc_weight: float = 5.0,
    local_max_nfev: int = 80,
):
    """
    Minimiza:

        R = [
            u''(x_j) + pi^2 sin(pi x_j),  para puntos interiores x_j
            bc_weight * u(0)
            bc_weight * u(1)
        ]

    y fitness = mean(R^2).

    IMPORTANTE:
    La solución exacta sin(pi*x) NO aparece aquí.
    """
    const_nodes = []

    try:
        expr = to_sympy(individual, const_nodes)
        d2_expr = sp.diff(expr, X, 2)

        const_symbols = [s for s, _ in const_nodes]

        u_func = sp.lambdify((X, *const_symbols), expr, modules="numpy")
        d2_func = sp.lambdify((X, *const_symbols), d2_expr, modules="numpy")

    except Exception:
        return 1e30, None

    def residual_vector(c):
        try:
            u2 = _as_vector(d2_func(x_phys, *c), len(x_phys))

            # Residual de la EDO:
            # u''(x) + pi^2 sin(pi x) = 0
            r_phys = u2 + (math.pi**2) * np.sin(math.pi * x_phys)

            # Condiciones de frontera:
            bc0 = float(np.asarray(u_func(0.0, *c)))
            bc1 = float(np.asarray(u_func(1.0, *c)))
            r_bc = bc_weight * np.array([bc0, bc1], dtype=float)

            r = np.concatenate([r_phys, r_bc])

            if not np.all(np.isfinite(r)):
                return np.full(len(x_phys) + 2, 1e6)

            return np.clip(r, -1e6, 1e6)

        except Exception:
            return np.full(len(x_phys) + 2, 1e6)

    # --------------------------------------------------------
    # Optimización local de constantes
    # --------------------------------------------------------
    c0 = np.array([node.value for _, node in const_nodes], dtype=float)

    if len(c0) > 0:
        try:
            # Igual que la idea del artículo: iniciar constantes en [-1,1]
            # pero dejar la optimización sin restricciones.
            result = least_squares(
                residual_vector,
                c0,
                method="lm" if (len(x_phys) + 2) >= len(c0) else "trf",
                max_nfev=local_max_nfev,
            )

            optimized = result.x

            # Actualización Lamarckiana: el individuo conserva sus
            # constantes localmente optimizadas.
            for (_, node), value in zip(const_nodes, optimized):
                node.value = float(value)

        except Exception:
            pass

    # Recalcular usando las constantes actualizadas
    c_final = np.array([node.value for _, node in const_nodes], dtype=float)
    residual = residual_vector(c_final)

    mse = float(np.mean(residual**2))

    # Rechazar resultados inválidos
    if not np.isfinite(mse):
        mse = 1e30

    substitutions = {symbol: node.value for symbol, node in const_nodes}
    expr_final = expr.subs(substitutions)

    return mse, expr_final


# ============================================================
# 5. Evolución genética
# ============================================================

def tournament_selection(population, fitnesses, tournament_size=3):
    ids = random.sample(range(len(population)), tournament_size)
    best_id = min(ids, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_id])


def evolve(
    seed: int = 67,
    population_size: int = 150,
    generations: int = 100,
    max_nodes: int = 12,
    max_depth_allowed: int = 6,
    crossover_rate: float = 0.50,
    mutation_rate: float = 0.50,
    n_collocation: int = 21,
    stop_mse: float = 1e-12,
):
    random.seed(seed)
    np.random.seed(seed)

    x_phys = make_collocation_points(seed, n_collocation)

    population = [
        random_tree(max_depth=random.randint(1, 4))
        for _ in range(population_size)
    ]

    best_individual = None
    best_mse = float("inf")
    history = []

    for generation in range(generations):
        fitnesses = []
        expressions = []

        for individual in population:
            mse, expr = evaluate_individual(individual, x_phys)
            fitnesses.append(mse)
            expressions.append(expr)

        generation_best_idx = int(np.argmin(fitnesses))
        generation_best_mse = fitnesses[generation_best_idx]

        if generation_best_mse < best_mse:
            best_mse = generation_best_mse
            best_individual = copy.deepcopy(population[generation_best_idx])

        history.append(best_mse)

        if generation % 5 == 0 or generation_best_mse <= stop_mse:
            print(
                f"Generación {generation:4d} | "
                f"mejor MSE = {best_mse:.3e} | "
                f"expr = {expression_with_values(best_individual)}"
            )

        if best_mse <= stop_mse:
            break

        # Elitismo: conservar el mejor individuo de la generación
        new_population = [copy.deepcopy(population[generation_best_idx])]

        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses)
            parent_b = tournament_selection(population, fitnesses)

            child_a = copy.deepcopy(parent_a)
            child_b = copy.deepcopy(parent_b)

            if random.random() < crossover_rate:
                child_a, child_b = crossover(
                    child_a,
                    child_b,
                    max_nodes=max_nodes,
                    max_depth_allowed=max_depth_allowed,
                )

            if random.random() < mutation_rate:
                child_a = mutate(
                    child_a,
                    max_nodes=max_nodes,
                    max_depth_allowed=max_depth_allowed,
                )

            if random.random() < mutation_rate:
                child_b = mutate(
                    child_b,
                    max_nodes=max_nodes,
                    max_depth_allowed=max_depth_allowed,
                )

            new_population.extend([child_a, child_b])

        population = new_population[:population_size]

    return best_individual, best_mse, np.array(history), x_phys


# ============================================================
# 6. Validación independiente
# ============================================================

def validate_and_plot(best_individual: Node, history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    expr = expression_with_values(best_individual)
    simplified = snap_known_constants(sp.simplify(expr))

    print("\nMejor expresión encontrada:")
    print("  ", simplified)

    # Verificación simbólica del residual:
    symbolic_residual = sp.simplify(
        sp.diff(simplified, X, 2)
        + sp.pi**2 * sp.sin(sp.pi * X)
    )

    print("\nResidual simbólico simplificado:")
    print("  ", symbolic_residual)

    print("\nCondiciones de frontera:")
    print("  u(0) =", sp.N(simplified.subs(X, 0), 16))
    print("  u(1) =", sp.N(simplified.subs(X, 1), 16))

    # Malla independiente para validación
    grid = np.linspace(0.0, 1.0, 401)
    model_func = sp.lambdify(X, simplified, modules="numpy")
    pred = np.asarray(model_func(grid), dtype=float)

    if pred.ndim == 0:
        pred = np.full_like(grid, float(pred))

    # SOLO PARA VALIDAR ESTE EJEMPLO:
    # la solución exacta NO se usó durante el entrenamiento.
    exact = np.sin(np.pi * grid)

    validation_mse = float(np.mean((pred - exact) ** 2))
    print(f"\nMSE contra la solución exacta (solo validación): {validation_mse:.3e}")

    # Gráfica solución
    plt.figure(figsize=(8, 5))
    plt.plot(grid, exact, label="Exacta: sin(pi x)")
    plt.plot(grid, pred, "--", label="GPSR")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solución analítica descubierta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solution.png", dpi=160)
    plt.close()

    # Gráfica convergencia
    plt.figure(figsize=(8, 5))
    plt.semilogy(np.arange(len(history)), np.maximum(history, 1e-30))
    plt.xlabel("Generación")
    plt.ylabel("Mejor MSE")
    plt.title("Convergencia")
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=160)
    plt.close()

    return validation_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--collocation", type=int, default=21)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    best, mse, history, x_phys = evolve(
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        max_nodes=args.max_nodes,
        n_collocation=args.collocation,
    )

    print("\n============================================")
    print("FIN DE LA EVOLUCIÓN")
    print("============================================")
    print("Mejor fitness physics-regularized:", mse)
    print("Puntos de colocación:", x_phys)

    validate_and_plot(best, history, Path(args.output))


if __name__ == "__main__":
    main()
