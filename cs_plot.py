import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

def aclarar_color(color_hex, factor=0.5):
    """
    Aclara un color hex (como '#BA84F0') tirando hacia blanco.
    El factor ∈ [0, 1]: cuanto más cerca de 1, más blanco.
    """
    r, g, b, a = to_rgba(color_hex)
    r_aclarado = r + (1 - r) * factor
    g_aclarado = g + (1 - g) * factor
    b_aclarado = b + (1 - b) * factor
    return (r_aclarado, g_aclarado, b_aclarado, a)

def prepare_data(data, sep, rs_a_col, rs_b_col, ref_a_col, ref_b_col):

    df = pd.read_csv(data, sep=sep)
    df = df[[rs_a_col, rs_b_col, ref_a_col, ref_b_col]]
    last_row = len(df)

    # Iniciamos listas vacías a las que añadiremos según datos crudos
    trials = [0]
    rs_a = [0]
    rs_b = [0]
    ref_a = [0]
    ref_b = [0]

    for row in df.itertuples(index=True):

        cur_trial = trials[-1]
        index = row.Index        
        cur_rs_a = row[1]
        cur_rs_b = row[2]
        cur_ref_a = row[3]
        cur_ref_b = row[4]

        if index == 0: #Revisar aquí, espero que no pongan más de una respuesta por registro
            
            rs_a[cur_trial] = cur_rs_a
            rs_b[cur_trial] = cur_rs_b
            
            if cur_ref_a > 0:

                ref_a[cur_trial] += cur_ref_a
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res a, ref b (0 por defecto) y acumula respuestas en B

                    trials.append(cur_trial+1)
                    ref_a.append(0)
                    rs_a.append(0)
                    ref_b.append(0)
                    rs_b.append(rs_b[-1])

            if cur_ref_b > 0:

                ref_b[cur_trial] += cur_ref_b
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res b, ref a (0 por defecto) y acumula respuestas en A

                    trials.append(cur_trial+1)
                    ref_b.append(0)
                    rs_b.append(0)
                    ref_a.append(0)
                    rs_a.append(rs_a[-1])

        if index > 0: # También hay que trabajar que quizás empiece con primera respuesta lleva a reforzador
                        
            if df.iloc[index][rs_a_col] > df.iloc[index-1][rs_a_col]: # Si en este mismo registro hay una respuesta más, ponlo

                rs_a[cur_trial] += 1
            
            if df.iloc[index][rs_b_col] > df.iloc[index-1][rs_b_col]: # Si en este mismo registro hay una respuesta más, ponlo

                rs_b[cur_trial] += 1

            if df.iloc[index][ref_a_col] > df.iloc[index-1][ref_a_col]: # "Si hay un nuevo reforzador en programa A"

                ref_a[cur_trial] += 1
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res a, ref b (0 por defecto) y acumula respuestas en B

                    trials.append(cur_trial+1)
                    ref_a.append(0)
                    rs_a.append(0)
                    ref_b.append(0)
                    rs_b.append(rs_b[-1])

            if df.iloc[index][ref_b_col] > df.iloc[index-1][ref_b_col]: # "Si hay un nuevo reforzador en programa B"

                ref_b[cur_trial] += 1
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res b, ref a (0 por defecto) y acumula respuestas en A

                    trials.append(cur_trial+1)
                    ref_b.append(0)
                    rs_b.append(0)
                    ref_a.append(0)
                    rs_a.append(rs_a[-1])

    trials = [trial + 1 for trial in trials]

    clean_df = pd.DataFrame(
        {
            "Trial": trials,
            "Responses A": rs_a,
            "Responses B":rs_b,
            "Reinforcement A": ref_a,
            "Reinforcement B": ref_b
        }
    )

    return clean_df
    
def plot_cs(clean_df, rs_a_col, rs_b_col, ref_a_col, ref_b_col, step=1, label_a="", label_b="", color_a="#D55E00", color_b="#0072B2"):

    df = clean_df.copy()

    # Crea el rango del eje X del gráfico
    rs_max = df[[rs_a_col, rs_b_col]].max().max()
    scale_x_axis = np.ceil(rs_max / step)
    x_axis_range = np.arange(int(-scale_x_axis*step), int(scale_x_axis*step+1), step=step)

    # Transformación para que salga a la izquierda
    df[rs_a_col] = df[rs_a_col] * -1

    # Diccionarios con los colores y su versión aclarada
    color_map_a = {0: aclarar_color(color_a), 1: color_a}
    color_map_b = {0: aclarar_color(color_b), 1: color_b}

    # Listado de color a usar en cada ensayo
    colors_a = [color_map_a[ref_a] for ref_a in df[ref_a_col]]
    colors_b = [color_map_b[ref_b] for ref_b in df[ref_b_col]]

    # Gráfico de los datos
    fig, ax = plt.subplots()
    bar_a = ax.barh(y=df["Trial"], width=df[rs_a_col], color=colors_a)
    bar_b = ax.barh(y=df["Trial"], width=df[rs_b_col], color=colors_b)
    
    ax.set_ylim(len(df)+1, 0)
    ax.set_xticks(x_axis_range, labels=abs(x_axis_range))
    
    ax.spines["left"].set_position(('data', 0))
    ax.spines[["right", "top"]].set_visible(False)

    ax.set_xlabel("Responses", fontdict={"weight": "bold"})

    ax.grid(axis="x", linestyle=":")

    # Crear handles personalizados para la leyenda con colores oscuros
    legend_handles = [
        Patch(facecolor=color_a, label=f"{label_a}"),
        Patch(facecolor=color_b, label=f"{label_b}")
    ]
    ax.legend(handles=legend_handles)
    
    plt.show()

    return fig, ax

def cumulative_records_setup(data, sep, rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col):

    df = pd.read_csv(data, sep=sep)
    df = df[[rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col]]
    
    return df

def cumulative_records_plot(clean_df, rs_a_col, rs_b_col, ref_a_col, ref_b_col, time_col, time_unit=None, label_a="", label_b="", color_a="#D55E00", color_b="#0072B2"):

    df = clean_df.copy()
    
    # Localiza reforzadores
    refs_a = df[df[ref_a_col] != df[ref_a_col].shift(1)]
    refs_b = df[df[ref_b_col] != df[ref_b_col].shift(1)]

    # Gráfico de los datos
    fig, ax = plt.subplots()
    ax.plot(time_col, rs_a_col, data=df, color=color_a)
    ax.plot(time_col, rs_b_col, data=df, color=color_b)    
    ax.scatter(refs_a[time_col], refs_a[rs_a_col], marker="x", color=color_a)
    ax.scatter(refs_b[time_col], refs_b[rs_b_col], marker="x", color=color_b)

    ax.spines[["right", "top"]].set_visible(False)

    # Títulos
    if time_unit != None:
        xlabel_unit = f" ({time_unit})"
    else:
        xlabel_unit = ""

    ax.set_xlabel(f"Time{xlabel_unit}", fontdict={"weight": "bold"})
    ax.set_ylabel("Responses", fontdict={"weight": "bold"})

    ax.set_xlim(0, df[time_col].max())
    ax.set_ylim(0, df[[rs_a_col, rs_b_col]].max().max())

    ax.grid(axis="y", linestyle=":")

    legend_handles = [
        Patch(facecolor=color_a, label=f"{label_a}"),
        Patch(facecolor=color_b, label=f"{label_b}")
    ]
    ax.legend(handles=legend_handles)
    
    plt.show()

    return fig, ax



clean_df = prepare_data(
    data="./Data/subject-1-13.csv",
    sep=";",
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh"
    )

plot_cs(
    clean_df = clean_df,
    rs_a_col = "Responses A",
    rs_b_col = "Responses B",
    ref_a_col = "Reinforcement A",
    ref_b_col = "Reinforcement B",
    step=50,
    label_a="Schedule A",
    label_b="Schedule B"
)

cumulative_records_df = cumulative_records_setup(
    data="./Data/subject-1-13.csv",
    sep=";",
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh",
    time_col = "current_time"
    )

cumulative_records_plot(
    clean_df = cumulative_records_df,
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh",
    time_col = "current_time",
    time_unit= "ms",
    label_a="Schedule A",
    label_b="Schedule B"
    )




'''
- Si los datos vienen con los reforzadores no acumulados, entonces no tengo que transformar buscar fila anterior para localizar ref
- Distintas posibilidades para escalar tiempo, quizás desde el pre-procesado antes del gráfico
- Eje X del cumulative record en ms crudos (600000). Ya tienes el parámetro time_unit, úsalo para convertir a minutos en el preprocesado, no solo en el label — time_unit="ms" con valores de 600k es difícil de leer de un vistazo.
'''