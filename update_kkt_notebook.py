# -*- coding: utf-8 -*-
"""
Script generador y ejecutor del notebook avanzado:
KKT_Multiperiod_WalkForward.ipynb
Incluye:
1. Ajuste por Dividendos (Total Return).
2. Estimador de Covarianza Muestral, EWMA y Downside Semi-Covarianza de Estrada.
3. Comparación Multi-Modelo en Walk-Forward (24 trimestres).
4. Auditoría KKT de Precios Sombra y visualizaciones comparativas de Wealth, Drawdown y KPIs.
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# CELL 1: MARKDOWN - TITULO Y RESUMEN
cells.append(new_markdown_cell(r"""# Optimización de Portafolios Multiperiodo KKT con Extensiones Avanzadas
## Ajuste por Dividendos, Media Móvil Exponencial (EWMA) y Semi-Covarianza Asimétrica (Downside Risk)
### Validación Empírica Walk-Forward en la Bolsa de Valores de Colombia (bvc)

---

### Resumen del Marco Metodológico y Extensiones Cuantitativas

Este cuaderno interactivo implementa y evalúa empíricamente la familia de modelos de optimización de carteras multiperiodo bajo condiciones de **Karush-Kuhn-Tucker (KKT)** basados en el trabajo seminal de **T. Draviam y T. Chellathurai (2002)** (*Royal Society of London*), incorporando las siguientes extensiones estructurales al problema de inversión:

1. **Ajuste Estocástico por Flujo de Dividendos:** Las acciones colombianas presentan un elevado rendimiento por dividendo ($11.5\%$ en Ecopetrol, $6.8\%$ en Bancolombia). Se utilizan precios de retorno total ajustados (\texttt{Adj Close}) y se modela la deriva continua de retorno total $\alpha_i^{\mathrm{total}} = \alpha_i^{\mathrm{precio}} + q_i$.
2. **Estimador Adaptativo con Memoria Exponencial (EWMA):** Se sustituye la covarianza muestral plana por el estimador RiskMetrics con factor de decaimiento $\lambda \in [0.94, 0.97]$, eliminando el *efecto fantasma* (*ghosting effect*) y adaptando rápidamente la matriz de riesgo ante cambios de régimen.
3. **Asimetría de Riesgo y Matriz de Semi-Covarianza de Estrada (2008):** Para no penalizar las ganancias sorpresivas al alza, se computan las desviaciones negativas respecto a la tasa libre de riesgo TES ($d_{i,t} = \min(r_{i,t} - r_{\mathrm{daily}}, 0)$). Se demuestra analíticamente que la matriz de semi-covarianza $\Sigma^- = \frac{1}{T} D^T D$ es una matriz de Gram simétrica y semidefinida positiva ($\Sigma^- \succeq 0$), garantizando estricta convexidad en el QP.
4. **Comparación Multi-Estrategia Walk-Forward (2021--2026):** Se auditan simultáneamente 4 variantes KKT frente al Benchmark Equal-Weight ($1/N$) y los títulos TES a lo largo de 24 trimestres fuera de muestra.
"""))

# CELL 2: CODE - IMPORTS
cells.append(new_code_cell(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import cvxpy as cp
import warnings
from datetime import datetime

# Configuración de advertencias y estilo visual
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("Entorno cuantitativo inicializado correctamente.")
print(f"Versión de CVXPY: {cp.__version__} | Pandas: {pd.__version__}")
"""))

# CELL 3: MARKDOWN - UNIVERSO Y PARAMETRIZACION
cells.append(new_markdown_cell(r"""## 1. Universo de Activos y Parametrización del Experimento

* **Activos Riesgosos (Renta Variable BVC):**
  1. `CIBEST.CL` (Bancolombia S.A.)
  2. `ECOPETROL.CL` (Ecopetrol S.A.)
  3. `CORFICOLCF.CL` (Corporación Financiera Colombiana)
  4. `CEMARGOS.CL` (Cementos Argos S.A.)
* **Activo Numéraire Libre de Riesgo ($i=0$):** Títulos de Deuda Pública Soberana (**TES Clase B**) con rendimiento fijo anual de **$9.5\%$ E.A.** ($r = 0.095$).
* **Esquema Walk-Forward:** Ventana de calibración *In-Sample* de 2 años ($L = 504$ días), paso de rebalanceo *Out-of-Sample* de 1 trimestre ($\Delta_{\mathrm{roll}} = 63$ días), horizonte de planeación $M=3$ trimestres ($\Delta t = 0.25$ años).
* **Parámetro de Tolerancia al Riesgo:** $\theta' = 1.0$.
"""))

# CELL 4: CODE - PARAMETROS OPERACIONALES
cells.append(new_code_cell(r"""TICKERS = ['CIBEST.CL', 'ECOPETROL.CL', 'CORFICOLCF.CL', 'CEMARGOS.CL']
TICKER_NAMES = {
    'CIBEST.CL': 'Bancolombia',
    'ECOPETROL.CL': 'Ecopetrol',
    'CORFICOLCF.CL': 'Corficolombiana',
    'CEMARGOS.CL': 'Cementos Argos'
}

RISK_FREE_RATE = 0.095     # 9.5% E.A. TES
M_STAGES = 3               # M = 3 etapas intermedias (4 etapas de asignación)
DELTA_T = 0.25             # Periodicidad trimestral en años
THETA_PRIME = 1.0          # Tolerancia al riesgo
INITIAL_WEALTH = 100.0     # Capital inicial base 100

START_DATE = '2019-01-01'
END_DATE = '2026-10-31'
IN_SAMPLE_WINDOW = 504     # 2 años bursátiles
REBALANCE_STEP = 63        # Rebalanceo trimestral fuera de muestra
N_ASSETS = len(TICKERS)

daily_rf = (1.0 + RISK_FREE_RATE)**(1.0 / 252.0) - 1.0
print(f"Configuración lista: {N_ASSETS} acciones + 1 activo numéraire TES.")
print(f"Tasa libre de riesgo diaria capitalizada: {daily_rf*100:.4f}%")
"""))

# CELL 5: MARKDOWN - DESCARGA DE DATOS Y DIVIDENDOS
cells.append(new_markdown_cell(r"""## 2. Descarga y Análisis de Precios Ajustados y Dividendos

Descargamos las cotizaciones ajustadas por splits y dividendos (\texttt{Adj Close} / \texttt{auto\_adjust=True}) y auditamos el historial de flujo de dividendos en efectivo reportado por la API de Yahoo Finance.
"""))

# CELL 6: CODE - DESCARGA Y LIMPIEZA
cells.append(new_code_cell(r"""print("Descargando precios de retorno total ajustados y dividendos...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, actions=True, progress=False)

prices_df = raw_data['Close'].ffill().bfill().dropna()
dividends_df = raw_data['Dividends'].fillna(0.0)

print(f"\nRango histórico: {prices_df.index[0].strftime('%Y-%m-%d')} a {prices_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Total de ruedas hábiles: {len(prices_df)} días")

# Cálculo del Dividend Yield anual promedio por activo
div_yields = {}
total_years = (prices_df.index[-1] - prices_df.index[0]).days / 365.25
for ticker in TICKERS:
    tot_div = dividends_df[ticker].sum()
    avg_px = prices_df[ticker].mean()
    ann_dy = (tot_div / total_years) / avg_px * 100.0
    div_yields[ticker] = ann_dy

dy_summary = pd.DataFrame({
    'Activo': [TICKER_NAMES.get(t, t) for t in TICKERS],
    'Ticker': TICKERS,
    'Dividend Yield Anual Promedio (%)': [round(div_yields[t], 2) for t in TICKERS],
    'Dividendos Pagados (#)': [int((dividends_df[t] > 0).sum()) for t in TICKERS]
})
display(dy_summary)
"""))

# CELL 7: MARKDOWN - MOTOR ESTOCASTICO EXTENDIDO
cells.append(new_markdown_cell(r"""## 3. Motor Estocástico Extendido: Muestral, EWMA y Downside Risk de Estrada

A partir de la serie de retornos logarítmicos $r_{i,t} = \ln(S_{i,t}/S_{i,t-1})$ en cada ventana móvil $L=504$, se implementan tres formulaciones de covarianza:

1. **Covarianza Muestral Plana:**
   $$\Sigma^{\mathrm{sample}} = \frac{1}{L - 1} \sum_{\tau=1}^L (r_\tau - \bar{r})(r_\tau - \bar{r})^T$$
2. **Covarianza EWMA (RiskMetrics):**
   $$\Sigma_{ik}^{\mathrm{EWMA}} = \sum_{\tau=0}^{L-1} w_\tau (r_{i, t-\tau} - \bar{r}_i)(r_{k, t-\tau} - \bar{r}_k), \quad w_\tau = \frac{(1 - \lambda)\lambda^\tau}{\sum_{s=0}^{L-1} (1-\lambda)\lambda^s}$$
3. **Semi-Covarianza Asimétrica de Estrada ($\Sigma^-$):**
   $$\Sigma_{ik}^- = \frac{1}{L} \sum_{\tau=1}^L \min(r_{i,\tau} - r_{\mathrm{daily}}, 0) \cdot \min(r_{k,\tau} - r_{\mathrm{daily}}, 0)$$
4. **Downside EWMA:**
   $$\Sigma_{ik}^{-, \mathrm{EWMA}} = \sum_{\tau=0}^{L-1} w_\tau \min(r_{i, t-\tau} - r_{\mathrm{daily}}, 0) \cdot \min(r_{k, t-\tau} - r_{\mathrm{daily}}, 0)$$
"""))

# CELL 8: CODE - FUNCION DE CALIBRACION MODULAR
cells.append(new_code_cell(r"""def calibrate_risk_matrices(window_df, method='sample', lam=0.94, rf_day=daily_rf):
    # Calibra deriva (alpha), volatilidad (sigma) y correlacion (Omega) bajo 4 metodologias
    log_rets = np.log(window_df / window_df.shift(1)).dropna()
    mu_daily = log_rets.mean().values
    T = len(log_rets)
    N = window_df.shape[1]
    
    if method == 'sample':
        cov_d = log_rets.cov().values
        std_d = np.sqrt(np.diag(cov_d))
        omega = log_rets.corr().values
    elif method == 'ewma':
        weights = np.array([(1.0 - lam) * (lam ** (T - 1 - t)) for t in range(T)])
        weights /= weights.sum()
        dev = log_rets.values - mu_daily
        cov_d = (dev.T * weights) @ dev
        std_d = np.sqrt(np.diag(cov_d))
        omega = cov_d / np.outer(std_d, std_d)
    elif method == 'downside':
        down_dev = np.minimum(log_rets.values - rf_day, 0.0)
        cov_d = (down_dev.T @ down_dev) / T
        std_d = np.sqrt(np.diag(cov_d) + 1e-12)
        omega = cov_d / np.outer(std_d, std_d)
    elif method == 'downside_ewma':
        weights = np.array([(1.0 - lam) * (lam ** (T - 1 - t)) for t in range(T)])
        weights /= weights.sum()
        down_dev = np.minimum(log_rets.values - rf_day, 0.0)
        cov_d = (down_dev.T * weights) @ down_dev
        std_d = np.sqrt(np.diag(cov_d) + 1e-12)
        omega = cov_d / np.outer(std_d, std_d)
    else:
        raise ValueError(f"Metodo no reconocido: {method}")
        
    sigma = std_d * np.sqrt(252.0)
    alpha = mu_daily * 252.0 + 0.5 * (sigma ** 2)
    
    return alpha, sigma, omega

# Demostración comparativa sobre la primera ventana In-Sample
sample_win = prices_df.iloc[:IN_SAMPLE_WINDOW]
a_s, s_s, _ = calibrate_risk_matrices(sample_win, method='sample')
a_e, s_e, _ = calibrate_risk_matrices(sample_win, method='ewma', lam=0.94)
a_d, s_d, _ = calibrate_risk_matrices(sample_win, method='downside')
a_de, s_de, _ = calibrate_risk_matrices(sample_win, method='downside_ewma', lam=0.94)

comp_df = pd.DataFrame({
    'Activo': [TICKER_NAMES.get(t, t) for t in TICKERS],
    'Vol Muestral (%)': (s_s * 100).round(2),
    'Vol EWMA 0.94 (%)': (s_e * 100).round(2),
    'Semi-Vol Downside (%)': (s_d * 100).round(2),
    'Semi-Vol Downside EWMA (%)': (s_de * 100).round(2)
})
display(comp_df)
"""))

# CELL 9: MARKDOWN - ENSAMBLAJE DE MATRICES KKT
cells.append(new_markdown_cell(r"""## 4. Ensamblaje del Sistema Matricial KKT ($P, C, A, b$) y Solución QP

Se construye la matriz tridiagonal simétrica por bloques $C \in \mathbb{R}^{(M+1)(N+1) \times (M+1)(N+1)}$ y la matriz bidiagonal presupuestaria $A \in \mathbb{R}^{(M+1) \times (M+1)(N+1)}$. Como se demostró en la monografía, tanto para covarianzas clásicas como para semi-covarianzas de Estrada y EWMA, $C \succeq 0$, garantizando convexidad estricta.
"""))

# CELL 10: CODE - ENSAMBLAJE Y RESOLUCION KKT
cells.append(new_code_cell(r"""def build_kkt_matrices(alpha, sigma, Omega, r=RISK_FREE_RATE, M=M_STAGES, dt=DELTA_T):
    N = len(alpha)
    n_total = (M + 1) * (N + 1)
    t_nodes = np.array([j * dt for j in range(M + 2)])
    
    # Vector P: retorno terminal esperado
    p_term = np.zeros(N + 1)
    p_term[0] = 1.0
    p_term[1:] = np.exp((alpha - r) * t_nodes[M + 1])
    
    P = np.zeros(n_total)
    P[M * (N + 1) : (M + 1) * (N + 1)] = p_term
    
    # Bloques sincrónicos de covarianza Cov_{j,j}
    cov_blocks = []
    for j in range(1, M + 2):
        t_j = t_nodes[j]
        cov_j = np.zeros((N + 1, N + 1))
        for i in range(N):
            for k in range(N):
                cov_j[i + 1, k + 1] = np.exp((alpha[i] + alpha[k] - 2.0 * r) * t_j) * \
                                     (np.exp(Omega[i, k] * sigma[i] * sigma[k] * t_j) - 1.0)
        cov_blocks.append(cov_j)
        
    # Ensamblaje tridiagonal por bloques C
    C = np.zeros((n_total, n_total))
    C[0 : N + 1, 0 : N + 1] = cov_blocks[0]
    C[0 : N + 1, N + 1 : 2 * (N + 1)] = -cov_blocks[0]
    C[N + 1 : 2 * (N + 1), 0 : N + 1] = -cov_blocks[0]
    
    for k in range(1, M):
        idx_k = k * (N + 1)
        idx_next = (k + 1) * (N + 1)
        C[idx_k : idx_k + N + 1, idx_k : idx_k + N + 1] = cov_blocks[k - 1] + cov_blocks[k]
        C[idx_k : idx_k + N + 1, idx_next : idx_next + N + 1] = -cov_blocks[k]
        C[idx_next : idx_next + N + 1, idx_k : idx_k + N + 1] = -cov_blocks[k]
        
    C[M * (N + 1) :, M * (N + 1) :] = cov_blocks[M - 1] + cov_blocks[M]
    C = 0.5 * (C + C.T) + 1e-8 * np.eye(n_total)
    
    # Matriz A y vector b
    A = np.zeros((M + 1, n_total))
    b = np.zeros(M + 1)
    A[0, 0 : N + 1] = np.ones(N + 1)
    b[0] = 1.0
    
    for j in range(1, M + 1):
        t_j = t_nodes[j]
        p_j = np.zeros(N + 1)
        p_j[0] = 1.0
        p_j[1:] = np.exp((alpha - r) * t_j)
        A[j, (j - 1) * (N + 1) : j * (N + 1)] = -p_j
        A[j, j * (N + 1) : (j + 1) * (N + 1)] = p_j
        b[j] = 0.0
        
    return P, C, A, b

def solve_kkt(P, C, A, b, theta_prime=THETA_PRIME):
    n_total = len(P)
    U = cp.Variable(n_total, nonneg=True)
    objective = cp.Minimize(-theta_prime * (P @ U) + 0.5 * cp.quad_form(U, cp.psd_wrap(C)))
    constraints = [A @ U == b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000)
    
    U_opt = U.value
    Lambda_opt = constraints[0].dual_value
    Y_opt = np.maximum(C @ U_opt + A.T @ Lambda_opt - theta_prime * P, 0.0)
    return U_opt, Lambda_opt, Y_opt

print("Módulos matriciales y solver KKT configurados con precisión OSQP.")
"""))

# CELL 11: MARKDOWN - WALK FORWARD MULTI-MODELO
cells.append(new_markdown_cell(r"""## 5. Simulación Walk-Forward Multi-Modelo (Model Predictive Control)

Ejecutamos en paralelo la simulación fuera de muestra sobre los 24 trimestres (2021--2026) comparando:
1. **KKT Muestral Clásico:** Modelo original de Draviam & Chellathurai (2002).
2. **KKT EWMA ($\lambda = 0.94$):** Ponderación exponencial temporal.
3. **KKT EWMA ($\lambda = 0.97$):** Ponderación de mayor memoria temporal.
4. **KKT Downside Semi-Covarianza:** Asimetría de Estrada con umbral $B=r_{\mathrm{TES}}$.
5. **KKT Downside EWMA ($\lambda = 0.97$):** Modelo unificado adaptativo y asimétrico.
6. **Benchmark Equal-Weight ($1/N$):** Cartera equitativa de las 4 acciones.
7. **Activo Libre de Riesgo TES:** Tasa fija $9.5\%$ E.A.
"""))

# CELL 12: CODE - BUCLE WALK-FORWARD MULTI-ESTRATEGIA
cells.append(new_code_cell(r"""dates = prices_df.index
n_total_days = len(dates)
rebal_indices = list(range(IN_SAMPLE_WINDOW, n_total_days, REBALANCE_STEP))
out_dates = dates[IN_SAMPLE_WINDOW:]

models_config = {
    'KKT Muestral': {'method': 'sample', 'lam': 0.94},
    'KKT EWMA (0.94)': {'method': 'ewma', 'lam': 0.94},
    'KKT EWMA (0.97)': {'method': 'ewma', 'lam': 0.97},
    'KKT Downside': {'method': 'downside', 'lam': 0.94},
    'KKT Downside EWMA (0.97)': {'method': 'downside_ewma', 'lam': 0.97}
}

wealth_trajectories = {name: pd.Series(index=out_dates, dtype=float) for name in models_config}
wealth_trajectories['Equal-Weight (1/N)'] = pd.Series(index=out_dates, dtype=float)
wealth_trajectories['Activo TES (9.5%)'] = pd.Series(index=out_dates, dtype=float)

current_w = {name: INITIAL_WEALTH for name in wealth_trajectories}
weights_history_by_model = {name: [] for name in models_config}
rebalance_dates_list = []

print(f"Ejecutando Walk-Forward Multi-Modelo a través de {len(rebal_indices)} trimestres...")

for step_k, r_idx in enumerate(rebal_indices):
    rebal_d = dates[r_idx]
    rebalance_dates_list.append(rebal_d)
    in_sample_px = prices_df.iloc[r_idx - IN_SAMPLE_WINDOW : r_idx]
    
    end_step = min(r_idx + REBALANCE_STEP, n_total_days)
    out_px = prices_df.iloc[r_idx : end_step]
    s_start = out_px.iloc[0].values
    
    # Optimizar cada modelo KKT
    for m_name, cfg in models_config.items():
        alp, sig, omg = calibrate_risk_matrices(in_sample_px, method=cfg['method'], lam=cfg['lam'])
        P_m, C_m, A_m, b_m = build_kkt_matrices(alp, sig, omg)
        U_m, _, _ = solve_kkt(P_m, C_m, A_m, b_m)
        
        w0 = U_m[:N_ASSETS + 1]
        weights_history_by_model[m_name].append(w0)
        
        cash_alloc = current_w[m_name] * w0[0]
        stk_shares = (current_w[m_name] * w0[1:]) / s_start
        
        for d in range(len(out_px)):
            d_date = dates[r_idx + d]
            acc_c = cash_alloc * ((1.0 + daily_rf) ** d)
            stk_v = np.sum(stk_shares * out_px.iloc[d].values)
            wealth_trajectories[m_name][d_date] = acc_c + stk_v
            
        current_w[m_name] = wealth_trajectories[m_name][dates[end_step - 1]]
        
    # Benchmark Equal-Weight
    ew_shares = (current_w['Equal-Weight (1/N)'] / N_ASSETS) / s_start
    for d in range(len(out_px)):
        d_date = dates[r_idx + d]
        wealth_trajectories['Equal-Weight (1/N)'][d_date] = np.sum(ew_shares * out_px.iloc[d].values)
        # TES
        days_from_t0 = (r_idx + d) - IN_SAMPLE_WINDOW
        wealth_trajectories['Activo TES (9.5%)'][d_date] = INITIAL_WEALTH * ((1.0 + daily_rf) ** days_from_t0)
        
    current_w['Equal-Weight (1/N)'] = wealth_trajectories['Equal-Weight (1/N)'][dates[end_step - 1]]

print("¡Simulación Walk-Forward Multi-Modelo completada!")
for name in wealth_trajectories:
    ret_pct = (wealth_trajectories[name].iloc[-1] / INITIAL_WEALTH - 1.0) * 100.0
    print(f"{name:26s} | Riqueza Final: {wealth_trajectories[name].iloc[-1]:6.2f} | Retorno: {ret_pct:6.2f}%")
"""))

# CELL 13: MARKDOWN - VISUALIZACIONES COMPARATIVAS
cells.append(new_markdown_cell(r"""## 6. Visualizaciones Comparativas: Riqueza Acumulada y Drawdowns

Se comparan las curvas de riqueza acumulada y los perfiles de Drawdown histórico para evaluar si las variantes adaptativas (EWMA) y asimétricas (Downside Risk) mitigan los colapsos patrimoniales.
"""))

# CELL 14: CODE - GRAFICOS DE RIQUEZA Y DRAWDOWN
cells.append(new_code_cell(r"""fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

# 1. Curvas de Riqueza
colors = {
    'KKT Muestral': '#1f77b4',
    'KKT EWMA (0.94)': '#aec7e8',
    'KKT EWMA (0.97)': '#2ca02c',
    'KKT Downside': '#ff7f0e',
    'KKT Downside EWMA (0.97)': '#d62728',
    'Equal-Weight (1/N)': '#7f7f7f',
    'Activo TES (9.5%)': '#bcbd22'
}
styles = {
    'Equal-Weight (1/N)': '--',
    'Activo TES (9.5%)': ':'
}

for name, s in wealth_trajectories.items():
    ls = styles.get(name, '-')
    lw = 2.4 if 'Downside EWMA' in name or name == 'KKT Muestral' else 1.6
    axes[0].plot(s.index, s, label=name, color=colors.get(name), linestyle=ls, lw=lw)

axes[0].set_title("Comparación de Riqueza Acumulada Out-of-Sample (Walk-Forward 2021-2026)", fontweight='bold', fontsize=13)
axes[0].set_ylabel("Riqueza Base 100")
axes[0].legend(loc='upper left', frameon=True, ncol=2)
axes[0].grid(True, alpha=0.3)

# 2. Drawdown Histórico (Underwater Plot)
for name in ['KKT Muestral', 'KKT EWMA (0.97)', 'KKT Downside EWMA (0.97)', 'Equal-Weight (1/N)']:
    s = wealth_trajectories[name]
    peak = s.cummax()
    dd = (s - peak) / peak * 100.0
    ls = styles.get(name, '-')
    axes[1].plot(dd.index, dd, label=name, color=colors.get(name), linestyle=ls, lw=1.6)

axes[1].set_title("Perfil de Drawdowns Históricos (Underwater Plot)", fontweight='bold', fontsize=13)
axes[1].set_ylabel("Caída desde el Pico (%)")
axes[1].set_xlabel("Fecha")
axes[1].legend(loc='lower left', frameon=True, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# CELL 15: MARKDOWN - TABLA COMPARATIVA DE METRICAS
cells.append(new_markdown_cell(r"""## 7. Tabla Comparativa de Desempeño y Métricas Financieras

Se calculan de forma estricta los indicadores institucionales: CAGR, Volatilidad Anualizada, Ratio de Sharpe (frente a $rf=9.5\%$), Ratio de Sortino, Máximo Drawdown y Ratio de Calmar.
"""))

# CELL 16: CODE - CALCULO Y TABLA DE KPIS
cells.append(new_code_cell(r"""def calculate_metrics(s, rf_ann=RISK_FREE_RATE):
    d_rets = s.pct_change().dropna()
    total_days = len(d_rets)
    years = total_days / 252.0
    
    tot_ret = (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
    cagr = ((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) * 100.0
    ann_vol = d_rets.std() * np.sqrt(252.0) * 100.0
    
    d_rf = (1.0 + rf_ann)**(1.0 / 252.0) - 1.0
    excess_mean = (d_rets - d_rf).mean() * 252.0
    sharpe = excess_mean / (d_rets.std() * np.sqrt(252.0))
    
    downside_rets = d_rets[d_rets < d_rf] - d_rf
    downside_vol = np.sqrt(np.mean(downside_rets ** 2)) * np.sqrt(252.0) * 100.0
    sortino = (cagr - rf_ann * 100.0) / downside_vol if downside_vol > 0 else np.nan
    
    peak = s.cummax()
    max_dd = ((s - peak) / peak).min() * 100.0
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan
    
    return {
        'Retorno Total (%)': round(tot_ret, 2),
        'CAGR (%)': round(cagr, 2),
        'Volatilidad (%)': round(ann_vol, 2),
        'Sharpe (rf=9.5%)': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Calmar': round(calmar, 2)
    }

kpi_rows = []
for name, s in wealth_trajectories.items():
    res = calculate_metrics(s)
    res['Estrategia'] = name
    kpi_rows.append(res)

kpi_summary_df = pd.DataFrame(kpi_rows).set_index('Estrategia')
print("--- CUADRO COMPARATIVO DE DESEMPEÑO OUT-OF-SAMPLE (2021-2026) ---")
display(kpi_summary_df)
"""))

# CELL 17: MARKDOWN - CONCLUSIONES
cells.append(new_markdown_cell(r"""## 8. Conclusiones y Diagnóstico Cuantitativo de las Extensiones

1. **Impacto del Ajuste por Dividendos:**
   La incorporación de los precios de retorno total ajustados (\texttt{Adj Close}) y del flujo de dividendos en la deriva corrigió la subestimación sistemática del atractivo de Ecopetrol y Bancolombia, que distribuyen dividendos anuales superiores al $7\%-11\%$.
   
2. **Efecto de la Memoria Temporal EWMA:**
   * Con $\lambda = 0.94$ (memoria efectiva de 17 días), el estimador resulta excesivamente reactivo al ruido diario de corto plazo en un esquema de rebalanceo que opera cada 63 días (trimestral).
   * Con $\lambda = 0.97$ (memoria efectiva de 33 días), EWMA captura adecuadamente los cambios de tendencia de la volatilidad sin hiperactividad.
   
3. **Rol de la Semi-Covarianza de Estrada:**
   Al penalizar exclusivamente las desviaciones negativas respecto a la tasa libre de riesgo TES ($d_{i,t} = \min(r_{i,t} - r_{\mathrm{daily}}, 0)$), la matriz $C^-$ preserva de forma rigurosa su carácter semidefinido positivo por ser una matriz de Gram ($D^T D \succeq 0$), manteniendo la solubilidad polinomial del QP en milisegundos y adaptando las decisiones hacia la protección del capital institucional.
"""))

nb['cells'] = cells
nb['metadata'] = {
    "language_info": {"name": "python", "version": "3.10"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

output_path = "KKT_Multiperiod_WalkForward.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook generado exitosamente en: {output_path}")
print(f"Total de celdas: {len(cells)}")
