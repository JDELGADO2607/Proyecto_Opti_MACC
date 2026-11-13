# Optimizacion Cuadratica Multiperiodo de Portafolios con Condiciones KKT
## Extension del Modelo de T. Draviam y T. Chellathurai (2002) en la Bolsa de Valores de Colombia (BVC)

Autor: Juan Mario Delgado Molina  
Curso: Optimizacion Lineal y Convexa / Investigacion de Operaciones  
Fecha: Noviembre 2026  

---

### Descripcion General del Proyecto

Este repositorio contiene la formulacion teorica, deduccion analitica primal-dual y validacion computacional fuera de muestra (2021--2026) de la extension del modelo de optimizacion de carteras intertemporal de T. Draviam y T. Chellathurai (2002) aplicado al mercado accionario colombiano (BVC).

El proyecto aborda las limitaciones del modelo clasico de Markowitz (estatico y de un solo periodo), resolviendo de forma conjunta la trayectoria de decisiones de inversion a lo largo de multiples fechas de rebalanceo hacia adelante. Incorpora el regimen de concentracion prudencial de las Administradoras de Fondos de Pensiones (AFP) y fiduciarias de Colombia (Decreto 2555 de 2010 de la Superintendencia Financiera de Colombia), que fija una cota maxima del 20% por emisor individual.

---

### Pilares Metodologicos y de Optimizacion

1. Naturaleza Determinista del Programa Cuadratico (QP):
   Aunque los precios evolucionan segun un Movimiento Browniano Geometrico Multivariado (GBMM), los momentos y covarianzas intertemporales se resuelven analiticamente en forma cerrada mediante la distribucion log-normal. La incertidumbre queda integrada previamente en el vector de precios esperados descontados P y la matriz de covarianza C, formulando un Programa Cuadratico determinista y continuo.

2. Condiciones de Karush-Kuhn-Tucker (KKT):
   * Necesidad: Demostrada mediante la Calificacion de Restricciones Lineales (LCQ) y el Lema de Farkas, valida para cualquier conjunto de cotas e igualdades afines.
   * Suficiencia: Demostrada algebraicamente mediante la desigualdad de convexidad de primer orden: F(U) >= F(U*) + grad(F(U*))^T (U - U*).
   * Unicidad Global: Garantizada por la semidefinicion positiva estricta de la matriz hessiana C (lambda_min(C) > 0).

3. Sistema Primal-Dual con Doble Conjunto Activo:
   * Multiplicadores de igualdad Lambda: Valor marginal de la riqueza descontada en cada etapa temporal, exhibiendo monotonía decreciente hacia la unidad en la liquidacion terminal: Lambda = [1.2899, 1.1708, 1.0631, 1.0000]^T.
   * Precios sombra de exclusion Y >= 0: Subsidio o deficit de retorno necesario para que un activo excluido entre al portafolio.
   * Precios sombra de concentracion Z >= 0: Costo de oportunidad generado por la restriccion regulatoria del 20%.

4. Resultados Financieros en la BVC (10 Acciones + TES):
   Al imponer el limite prudencial del 20%, el optimizador satura exactamente 5 emisores en la frontera reglamentaria, mitigando la sobreconcentracion del modelo base y reduciendo el Max Drawdown de -48.84% a -27.07%, con una caida de la volatilidad anualizada de 32.27% a 19.74%.

5. Evaluacion de la Varianza Constante vs Estimadores Dinamicos:
   Se analiza empíricamente la falla del supuesto de varianza constante (clustering de volatilidad en series como Ecopetrol) y se evaluan filtros adaptativos (EWMA lambda=0.97, Downside Semi-Covarianza de Estrada y GJR-GARCH asimetrico) dentro de un esquema de Control Predictivo por Modelo (Walk-Forward MPC).

---

### Estructura del Repositorio

* `KKT_G_Markowitz.pdf`: Monografia completa del proyecto con la formulacion matematica, derivaciones KKT y tablas fuera de muestra.
* `KKT_G_Markowitz.tex`: Codigo fuente LaTeX de la monografia.
* `KKT_Optimizacion_Beamer.pdf`: Presentacion formal de 30 diapositivas para el curso de optimizacion, detallando la conexion con la dualidad de Programacion Lineal, demostraciones KKT y resultados empiricos.
* `KKT_Optimizacion_Beamer.tex`: Codigo fuente LaTeX de la presentacion Beamer.
* `KKT_Multiperiod_WalkForward.ipynb`: Jupyter Notebook interactivo con la descarga de datos reales de Yahoo Finance, calibracion parametrica, optimizacion con CVXPY (solvers OSQP y Clarabel), auditoria KKT y graficos de areas apiladas de asignacion dinamica.
* `Entregables/`: Directorio sincronizado que contiene los archivos finales compilados, figuras vectoriales y respaldos.
* `Paper_Original/`: Copia del articulo de referencia de T. Draviam y T. Chellathurai (2002).
* `grafica_retornos_varianza_constante.pdf` / `.png`: Visualizacion de la serie temporal de retornos y bandas estaticas vs dinamicas.
* `build_10_asset_notebook.py`: Script para generacion y ejecucion headless del cuaderno de investigacion.

---

### Requisitos de Ejecucion

Entorno Python recomendado (>= 3.10):
```bash
pip install numpy pandas matplotlib seaborn yfinance cvxpy osqp clarabel arch nbformat nbclient
```

Compilacion de documentos LaTeX:
Requiere una distribucion TeX (TeX Live o MiKTeX) con `pdflatex`.
```bash
pdflatex KKT_G_Markowitz.tex
pdflatex KKT_Optimizacion_Beamer.tex
```
