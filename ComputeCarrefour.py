import pyodbc
import pandas as pd
import datetime
import numpy as np
from math import factorial
import statistics as stats
import matplotlib.pyplot as plt
from datetime import timedelta
import csv

user = "D"
mode = 3

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# Takes a row and returns the week number corresponding to the date in the DATE column
def func(row):
    return(row["DATE"].isocalendar()[1])

# Takes a row and returns the trend value corresponding to the value in the cilumn "volumen"
def calc_trend(row):
    return(1.0/( row["volumen"]/(1.0/52.0)))

#Takes a row and returns KL_DETREND or 0 corresponding to the value in promo columns
def ventapromo(row):
    if row["Animacion 1"]==0:
        return 0
    else: return row["KL_DETREND"]

#Takes a row and returns EUROS_DETREND or 0 corresponding to the value in promo columns
def eurospromo(row):
    if row["Animacion 1"]==0:
        return 0
    else: return row["EUROS_DETREND"]

def ispromo(row):
    if row["Animacion 1"]==0:
        return 0
    else: return "P"

def replace(row):
    if row["KL_DETREND"]<=0.1:
        return row["KL_DETREND"]
    else: return row["BASELINE"]

def changepromo(df_total, status, canib, date):
    if status!="P":
        print("DISTINTO DE P")
        aux_df = df_total[(df_total["Grupo canibalizacion"] == canib) & (
        df_total["DATE"] == date)]
        print("CREADO DF")
        if "P" in aux_df["STATUS_PROMO"].values:
            return "C"

# Establish connection to SAP HANA server
cnxn = pyodbc.connect('Driver=HDBODBC;SERVERNODE=172.31.100.155:30041;UID=SAPEP01;PWD=EfiProm2017')

# Gets a cursor on the server to perform query
cursor = cnxn.cursor()

# Initialize results list to an empty list
entries = []

# Gets all the combinations "Material"+"Enseña"+"Central Data"+"Familia APO"
cursor.execute(
    'SELECT DISTINCT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZCDATA","_BIC_ZFAMAPO"  FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1"')

# Store all the combinations into the list
for row in cursor.fetchall():
    entries.append(row)

print(entries)


interp = False

############### This part of the code has to be executed for each "Material"+"Enseña"+"Control Data"+"Familia APO" combination !!!!!!!!

# To test. In production, this is derived from the "Material"+"Enseña"+"Control Data"+"Familia APO" that is processed
cpt = 0
df_total=None

# We read promotion file and make a new dataframe to use the function "join" in order to calculate promos

if user == "D" and mode == 1:
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1710_II.xlsx"
elif user == "M" and mode == 1:
    promo_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1710_II.XLSX"
elif user == "D" and mode == 2:
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_ECI_2015_1710_VersIII.XLSX"
elif user == "M" and mode == 2:
    promo_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\PROMOCIONES_ECI_2015_1710_VersIII.XLSX"

promo = pd.read_excel(promo_file)
#print(promo.duplicated())
promo=promo.drop_duplicates(subset=["COD ENSEÑA", "CODIGO CLIENTE", "Fecha inicio folleto","Fecha fin folleto"," CODFamilia apo"])
promo=promo.reset_index(drop=True)
print(len(promo))
print(promo)

df_promo=pd.DataFrame(columns=["ENS","FAMAPO","DATE","Animacion 1", "Animacion 2", "Animacion 3", "TEMATICA","Abreviatura accion", "CDATA", "Codigo unico"])
data_matrix = []

cont=0
days_before=0
days_after=0
for row_promo in promo.values:
    #print(row_promo["Animacion 1"])
    print(row_promo[0])
    last_date=row_promo[4]+timedelta(days=days_after)
    first_date=row_promo[3]-timedelta(days=days_before)
    diff=last_date-(first_date-timedelta(days=days_before))
    data_matrix.append([row_promo[2], row_promo[7], first_date,
                        row_promo[9],row_promo[10],row_promo[11],
                        row_promo[14],row_promo[9], int(row_promo[1]), row_promo[0]])
    #df_promo.values[cont]=[row_promo[2], row_promo[7], first_date,
    #                    row_promo[9],row_promo[10],row_promo[11],
    #                    row_promo[14],row_promo[9], int(row_promo[1])]
    cont+=1
    for j in range(1,diff.days):
        #print(j)
        #print(row_promo[3]+j+1)
        #print(type(row_promo[3]))
        d=timedelta(days=j)
        data_matrix.append([row_promo[2], row_promo[7], first_date+d,
                              row_promo[9], row_promo[10], row_promo[11],
                              row_promo[14], row_promo[9], int(row_promo[1]), row_promo[0]])
     #   df_promo.loc[cont] = [row_promo[2], row_promo[7], first_date+d,
     #                         row_promo[9], row_promo[10], row_promo[11],
     #                         row_promo[14], row_promo[9], int(row_promo[1])]
        cont+=1
    data_matrix.append([row_promo[2] , row_promo[7], last_date+timedelta(days=days_after),
                        row_promo[9], row_promo[10], row_promo[11],
                        row_promo[14], row_promo[9], int(row_promo[1]), row_promo[0]])
    #df_promo.loc[cont]=[row_promo[2] , row_promo[7], last_date+timedelta(days=days_after),
    #                    row_promo[9], row_promo[10], row_promo[11],
    #                    row_promo[14], row_promo[9], int(row_promo[1])]
    cont+=1
    print(cont)
    #print(df_promo)
df_promo = pd.DataFrame(data_matrix)
df_promo.columns = ["ENS","FAMAPO","DATE","Animacion 1", "Animacion 2", "Animacion 3", "TEMATICA","Abreviatura accion", "CDATA", "Codigo unico"]

print(df_promo)

# We read the seasonality file
station_file = "C:\\Users\\tr5568\\Desktop\\DAYANA\\CAPSA\\Tabla_estacionalidad fichero carga.xlsx"
station = pd.read_excel(station_file, 1)
for ent in entries:
    if ent[3] in ["340", "341","360", "366","470","471"] and ent[1] == "Z5E99K":
    #if ent[1]=="Z5E99K" and ent[3]!="111":
    #if ent[3]!="111":
    #if ent[3] =="550" and ent[1] == "Z5E99K" and ent[0]=="000000000000014129" and ent[2]=="0000121062":
        print("VALOR DE SFAPO: ")
        print(str(ent[3]))
        if(str(ent[3])==''): SFAPO=0
        else: SFAPO = int(ent[3])

        #### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
        ## This part of the query stays fixed
        #query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
        query = 'SELECT "_BIC_ZCDATA","_BIC_ZMARCA2","_BIC_ZMATERIA2","_BIC_ZSECCION2","_BIC_ZSUBSEC2", "CENTRALDATAT","MARCAT","MATERIALT","SECCIONT","SUBSECCIONT","_BIC_ZDESTMER","DESTINATARIOT1", "_BIC_ZENSENA","_BIC_ZCODPOST","DESTINATARIOT2", "CALMONTH","CALWEEK","ZMM","ZAAAA",sum("_BIC_ZCOUNTER") AS "_BIC_ZCOUNTER",sum("_BIC_ZIMPPVP") AS "_BIC_ZIMPPVP",sum("_BIC_ZUNIDFRA") AS "_BIC_ZUNIDFRA",sum("_BIC_ZVMKLESTA") as "_BIC_ZVMKLESTA" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSD_DE03" GROUP BY "_BIC_ZCDATA","_BIC_ZMARCA2", "_BIC_ZMATERIA2","_BIC_ZSECCION2","_BIC_ZSUBSEC2","CENTRALDATAT","MARCAT","MATERIALT","SECCIONT","SUBSECCIONT","_BIC_ZDESTMER","DESTINATARIOT1", "_BIC_ZENSENA","_BIC_ZCODPOST","DESTINATARIOT2","CALMONTH","CALWEEK", "ZMM","ZAAAA"'
        query = query + '"ZFECHA" >= 20170101 AND "_BIC_ZENSENA2" NOT IN (\'Z5E005\',\'Z5E008\',\'Z5E013\',\'Z5E018\') AND '
        ## Here goes the adaptation: replace the hard coded values with the variable of the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
        ## These values are hard coded to test
        query = query + '"_BIC_ZENSENA2" = \''+ent[1]+'\' AND "_BIC_ZFAMAPO"=\''+ent[3]+'\' AND "_BIC_ZMATERIAL"=\''+ent[0]+'\' AND "_BIC_ZCDATA"=\''+ent[2]+'\' '
        ## This part of the query stays fixed
        #query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZDESMER70","ZFECHA" '
        query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZCDATA","ZFECHA" '
        query = query + 'ORDER BY "ZFECHA"'

        print("SQL Query: "+query)

        # We run the query to get all the entries for a particular "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
        cursor.execute(query)