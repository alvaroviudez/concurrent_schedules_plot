import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba


def aclarar_color(color_hex, factor=0.4):
    """
    Aclara un color hex (como '#BA84F0') tirando hacia blanco.
    El factor ∈ [0, 1]: cuanto más cerca de 1, más blanco.
    """
    r, g, b, a = to_rgba(color_hex)
    r_aclarado = r + (1 - r) * factor
    g_aclarado = g + (1 - g) * factor
    b_aclarado = b + (1 - b) * factor
    return (r_aclarado, g_aclarado, b_aclarado, a)

'''
Supongamos aquí datos de:
Programa A: RF5
Programa B: RF10
'''


# Datos de prueba en formato crudo

# Datos de prueba en formato final
trials = [1, 2, 3, 4, 5]
rs_a = [5, 5, 2, 4, 5] # Aquí ya son acumulativas hasta el ref: pasar de 2 a 4 significa que dio 2 más
rs_b = [0, 0, 10, 10, 3]
ref_a = [1, 1, 0, 0, 1]
ref_b = [0, 0, 1, 1, 0]

df = pd.DataFrame(
    {
        "Trial": trials,
        "Responses A": rs_a,
        "Responses B":rs_b,
        "Reinforcement A": ref_a,
        "Reinforcement B": ref_b
    }
)

df["Responses A"] = df["Responses A"] * -1 # Transformación para que salga a la izquierda

def plot_cs(df, color_a="#F08182", color_b="#818CF0"):

    # Crea el rango del eje X del gráfico
    rs_max = df[["Responses A", "Responses B"]].max().max()
    x_axis_range = np.arange(-rs_max, rs_max+1)

    # Diccionarios con los colores y su versión aclarada
    color_map_a = {0: aclarar_color(color_a), 1: color_a}
    color_map_b = {0: aclarar_color(color_b), 1: color_b}

    # Listado de color a usar en cada ensayo
    colors_a = [color_map_a[ref_a] for ref_a in df["Reinforcement A"]]
    colors_b = [color_map_b[ref_b] for ref_b in df["Reinforcement B"]]

    # Gráfico de los datos
    fig, ax = plt.subplots()
    ax.barh(y=df["Trial"], width=df["Responses A"], color=colors_a)
    ax.barh(y=df["Trial"], width=df["Responses B"], color=colors_b)
    
    ax.set_ylim(len(df)+1, 0)
    ax.set_xticks(x_axis_range, labels=abs(x_axis_range))
    
    ax.spines["left"].set_position(('data', 0))
    ax.spines[["right", "top"]].set_visible(False)
    
    plt.show()

    return None

plot_cs(df)
