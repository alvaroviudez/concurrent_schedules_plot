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

    for index, row in df.iloc[:3].iterrows():

        print(trials)

        cur_trial = trials[-1]

        if index == 0: #Revisar aquí, espero que no pongan más de una respuesta por registro
            rs_a[cur_trial] = row[rs_a_col]
            rs_b[cur_trial] = row[rs_b_col]
            
            if row[ref_a_col] > 0:

                ref_a[cur_trial] += row[ref_a_col]
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res a, ref b (0 por defecto) y acumula respuestas en B

                    trials.append(cur_trial+1)
                    ref_a.append(0)
                    rs_a.append(0)
                    ref_b.append(0)
                    rs_b.append(rs_b[-1])

            if row[ref_b_col] > 0:

                ref_b[cur_trial] += row[ref_b_col]
                
                if index + 1 < last_row: # Si aún quedan datos, resetea ref y res b, ref a (0 por defecto) y acumula respuestas en A

                    trials.append(cur_trial+1)
                    ref_b.append(0)
                    rs_b.append(0)
                    ref_a.append(0)
                    rs_a.append(rs_b[-1])

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
                    rs_a.append(rs_b[-1])

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

    clean_df["Responses A"] = clean_df["Responses A"] * -1 # Transformación para que salga a la izquierda
    
    return clean_df
    




clean_df = prepare_data(
    data="./Data/subject-1-13.csv",
    sep=";",
    rs_a_col = "respFi",
    rs_b_col = "respCh",
    ref_a_col = "reinfFi",
    ref_b_col = "reinfCh"
    )


'''
Supongamos aquí datos de:
Programa A: RF5
Programa B: RF10

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

'''

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
