import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t

# Dados fornecidos
data_threads = {
    "Nº de Threads": [1, 2, 5, 10, 15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    "Total Time Taken (s)": [216, 190, 219, 226, 201, 206, 180, 171, 163, 157, 157, 152, 156, 151, 149, 150, 156],
    "Memory Used by Threads (MB)": [163, 176, 167, 196, 202, 182, 213, 187, 247, 207, 256, 257, 262, 344, 307, 378, 305]
}

data_clients = {
    "Nº de Clientes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Total Time Taken (s)": [3, 6, 10, 19, 42, 65, 97, 141, 163, 228],
    "Memory Used (MB)": [6, 8, 12, 15, 17, 18, 20, 30, 39, 44]
}

df_threads = pd.DataFrame(data_threads)
df_clients = pd.DataFrame(data_clients)

# Função para ajuste exponencial
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Função para calcular intervalo de confiança
def confidence_interval(popt, pcov, alpha=0.05):
    n_params = len(popt)
    dof = max(0, len(df_threads) - n_params)  # graus de liberdade
    t_val = t.ppf(1 - alpha / 2, dof)  # valor crítico de t
    errors = np.sqrt(np.diag(pcov))  # desvios padrão dos parâmetros
    ci = t_val * errors  # intervalo de confiança
    return ci

# Ajuste para 'Nº de Threads vs Total Time Taken'
popt_time_threads, pcov_time_threads = curve_fit(exponential_func, df_threads["Nº de Threads"], df_threads["Total Time Taken (s)"])
ci_time_threads = confidence_interval(popt_time_threads, pcov_time_threads)

# Ajuste para 'Nº de Threads vs Memory Used by Threads'
popt_memory_threads, pcov_memory_threads = curve_fit(exponential_func, df_threads["Nº de Threads"], df_threads["Memory Used by Threads (MB)"])
ci_memory_threads = confidence_interval(popt_memory_threads, pcov_memory_threads)

# Ajuste para 'Nº de Clientes vs Total Time Taken'
popt_time_clients, pcov_time_clients = curve_fit(exponential_func, df_clients["Nº de Clientes"], df_clients["Total Time Taken (s)"])
ci_time_clients = confidence_interval(popt_time_clients, pcov_time_clients)

# Ajuste para 'Nº de Clientes vs Memory Used'
popt_memory_clients, pcov_memory_clients = curve_fit(exponential_func, df_clients["Nº de Clientes"], df_clients["Memory Used (MB)"])
ci_memory_clients = confidence_interval(popt_memory_clients, pcov_memory_clients)

# Impressão dos resultados
print("Regressão Exponencial para Threads:")
print(f"Nº de Threads vs Total Time Taken (s): a = {popt_time_threads[0]:.3f}, b = {popt_time_threads[1]:.3f}, CI = ±{ci_time_threads}")
print(f"Nº de Threads vs Memory Used by Threads (MB): a = {popt_memory_threads[0]:.3f}, b = {popt_memory_threads[1]:.3f}, CI = ±{ci_memory_threads}")

print("\nRegressão Exponencial para Clientes:")
print(f"Nº de Clientes vs Total Time Taken (s): a = {popt_time_clients[0]:.3f}, b = {popt_time_clients[1]:.3f}, CI = ±{ci_time_clients}")
print(f"Nº de Clientes vs Memory Used (MB): a = {popt_memory_clients[0]:.3f}, b = {popt_memory_clients[1]:.3f}, CI = ±{ci_memory_clients}")

# Exemplo de cálculo do R² para ajustar regressão exponencial
def r_squared(y, y_pred):
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# Cálculo de R²
r2_time_threads = r_squared(df_threads["Total Time Taken (s)"], exponential_func(df_threads["Nº de Threads"], *popt_time_threads))
r2_memory_threads = r_squared(df_threads["Memory Used by Threads (MB)"], exponential_func(df_threads["Nº de Threads"], *popt_memory_threads))
r2_time_clients = r_squared(df_clients["Total Time Taken (s)"], exponential_func(df_clients["Nº de Clientes"], *popt_time_clients))
r2_memory_clients = r_squared(df_clients["Memory Used (MB)"], exponential_func(df_clients["Nº de Clientes"], *popt_memory_clients))

# Impressão do R²
print("\nR² para Threads:")
print(f"Nº de Threads vs Total Time Taken (s): R² = {r2_time_threads:.3f}")
print(f"Nº de Threads vs Memory Used by Threads (MB): R² = {r2_memory_threads:.3f}")

print("\nR² para Clientes:")
print(f"Nº de Clientes vs Total Time Taken (s): R² = {r2_time_clients:.3f}")
print(f"Nº de Clientes vs Memory Used (MB): R² = {r2_memory_clients:.3f}")
