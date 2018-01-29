import pyodbc
import pandas as pd
import datetime
import numpy as np
from math import factorial
import statistics as stats
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage.filters import maximum_filter1d
import peakutils

file_name = "PROMOCIONES_EROSKI_LYB_DDLL_2015_1712.xlsx"
user = "M"

if user == "M":
    prefix = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\"
elif user == "D":
    prefix = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\"
elif user == "S":
    prefix = "C:\\Datos analisis\\"





promo = pd.read_excel(prefix+file_name)
# print(promo)
# print(len(promo))
# print(promo.duplicated())
promo = promo.drop_duplicates(
    subset=["COD ENSEÑA", "CODIGO CLIENTE", "Fecha inicio folleto", "Fecha fin folleto", " CODFamilia apo"])
promo = promo.reset_index(drop=True)
# print(len(promo))
# print(promo)

df_promo = pd.DataFrame(
    columns=["ENS", "FAMAPO", "DATE", "Animacion 1", "Animacion 2", "Animacion 3", "TEMATICA", "Abreviatura accion",
             "CDATA", "Codigo unico"])
data_matrix = []

cont = 0
days_before = 0
days_after = 0
for row_promo in promo.values:
    # print(row_promo["Animacion 1"])
    # print(row_promo[0])
    last_date = row_promo[4] + timedelta(days=days_after)
    first_date = row_promo[3] - timedelta(days=days_before)
    diff = last_date - (first_date - timedelta(days=days_before))
    entry = {'ENS':str(row_promo[2]), 'FAMAPO':int(row_promo[7]), 'DATE':first_date, 'Animacion 1':str(row_promo[9]), 'Animacion 2':str(row_promo[10]), 'Animacion 3':str(row_promo[11]), 'TEMATICA':str(row_promo[14]),
                    'Abreviatura accion':str(row_promo[8]), 'CDATA':int(row_promo[1]), 'Codigo unico':str(row_promo[0])}
    # df_promo.values[cont]=[row_promo[2], row_promo[7], first_date,
    #                    row_promo[9],row_promo[10],row_promo[11],
    #                    row_promo[14],row_promo[9], int(row_promo[1])]
    cont += 1
    for j in range(1, diff.days):
        # print(j)
        # print(row_promo[3]+j+1)
        # print(type(row_promo[3]))
        d = timedelta(days=j)

        entry = {'ENS': str(row_promo[2]), 'FAMAPO': int(row_promo[7]), 'DATE': first_date + d,
                 'Animacion 1': str(row_promo[9]), 'Animacion 2': str(row_promo[10]), 'Animacion 3': str(row_promo[11]),
                 'TEMATICA': str(row_promo[14]),
                 'Abreviatura accion': str(row_promo[8]), 'CDATA': int(row_promo[1]), 'Codigo unico': str(row_promo[0])}

        data_matrix.append([row_promo[2], row_promo[7], first_date + d,
                            row_promo[9], row_promo[10], row_promo[11],
                            row_promo[14], row_promo[8], int(row_promo[1]), row_promo[0]])
        #   df_promo.loc[cont] = [row_promo[2], row_promo[7], first_date+d,
        #                         row_promo[9], row_promo[10], row_promo[11],
        #                         row_promo[14], row_promo[9], int(row_promo[1])]
        cont += 1
        print(entry)
    entry = {'ENS': str(row_promo[2]), 'FAMAPO': int(row_promo[7]), 'DATE': last_date + timedelta(days=days_after),
             'Animacion 1': str(row_promo[9]), 'Animacion 2': str(row_promo[10]), 'Animacion 3': str(row_promo[11]),
             'TEMATICA': str(row_promo[14]),
             'Abreviatura accion': str(row_promo[8]), 'CDATA': int(row_promo[1]), 'Codigo unico': str(row_promo[0])}
    data_matrix.append([row_promo[2], row_promo[7], last_date + timedelta(days=days_after),
                        row_promo[9], row_promo[10], row_promo[11],
                        row_promo[14], row_promo[8], int(row_promo[1]), row_promo[0]])
    # df_promo.loc[cont]=[row_promo[2] , row_promo[7], last_date+timedelta(days=days_after),
    #                    row_promo[9], row_promo[10], row_promo[11],
    #                    row_promo[14], row_promo[9], int(row_promo[1])]
    cont += 1
    # print(cont)
    # print(df_promo)
df_promo = pd.DataFrame(data_matrix)
df_promo.columns = ["ENS", "FAMAPO", "DATE", "Animacion 1", "Animacion 2", "Animacion 3", "TEMATICA",
                    "Abreviatura accion", "CDATA", "Codigo unico"]
df_promo = df_promo.drop_duplicates(subset=["ENS", "CDATA", "DATE", "FAMAPO"])
df_promo = df_promo.reset_index(drop=True)

# print(df_promo)