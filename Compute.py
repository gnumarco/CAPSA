import pyodbc
import pandas as pd
import datetime
import numpy as np
from math import factorial
import statistics as stats
#import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage.filters import maximum_filter1d
import peakutils

import csv

closed_list = [446, 5002, 5004, 5005, 5011, 5012, 5013, 5015, 5018, 5042, 5058, 5073, 5081, 5094, 5123, 5126, 5162,
               5302, 5317, 5324, 5326, 5327, 5474, 5728, 5740, 5741, 5755, 5788, 7425, 7449, 7450]

mode = 2  # Eroski
#mode = 2 #ECI
#mode = 3 # CRF
user = "M"

# mode_baseline 2 is means by week days
mode_baseline = 2


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
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    print(y[0])
    print(np.abs(y[1:half_window + 1][::-1] - y[0]))
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of flat, hanning, hamming, bartlett, blackman")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def baseline_als(y, lam=1000, p=0.01, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# Takes a row and returns the week number corresponding to the date in the DATE column
def func(row):
    return (row["DATE"].isocalendar()[1])


# Takes a row and returns the trend value corresponding to the value in the cilumn "volumen"
def calc_trend(row):
    return (1.0 / (row["volumen"] / (1.0 / 52.0)))


# Takes a row and returns KL_DETREND or 0 corresponding to the value in promo columns
def ventapromo(row):
    if row["Animacion 1"] == 0:
        return 0
    else:
        return row["KL_DETREND"]


# Takes a row and returns EUROS_DETREND or 0 corresponding to the value in promo columns
def eurospromo(row):
    if row["Animacion 1"] == 0:
        return 0
    else:
        return row["EUROS_DETREND"]


def ispromo(row):
    if row["Animacion 1"] == 0:
        return 0
    else:
        return "P"


def replace(row):
    if row["KL_DETREND"] <= 0.1:
        return row["KL_DETREND"]
    else:
        return row["BASELINE"]


def changepromo(df_total, status, canib, date):
    if status != "P":
        print("DISTINTO DE P")
        aux_df = df_total[(df_total["Grupo canibalizacion"] == canib) & (
                df_total["DATE"] == date)]
        print("CREADO DF")
        if "P" in aux_df["STATUS_PROMO"].values:
            return "C"


def slicing(BASELINE, deg=3):
    slicing = 90
    part = len(BASELINE) // slicing
    rest = len(BASELINE) % slicing
    print("PART")
    print(part)
    print("REST")
    print(rest)
    new_baseline = []
    for zz in range(0, part):
        tmp_baseline = BASELINE[zz * slicing:(zz * slicing) + slicing]
        print(tmp_baseline)
        tmp_baseline = peakutils.baseline(tmp_baseline, deg=deg, max_it=1000, tol=0.000001)
        new_baseline.extend(tmp_baseline)
    new_baseline.extend(peakutils.baseline(BASELINE[part:part + rest], deg=deg, max_it=1000, tol=0.000001))

    print("New Baseline")
    print(len(BASELINE))
    print(len(new_baseline))
    print(new_baseline)
    return new_baseline


def max_filter1d_valid(a, W):
    b = []
    hW = W // 2
    for i in range(hW, len(a) - hW):
        print(a[i - hW:i + hW])
        max = np.amax(a[i - hW:i + hW])
        print(max)
        b.append(max)
    return b


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


# Establish connection to SAP HANA server
cnxn = pyodbc.connect('Driver=HDBODBC;SERVERNODE=172.31.100.155:30041;UID=SAPEP01;PWD=EfiProm2017')

# Gets a cursor on the server to perform query
cursor = cnxn.cursor()

# Initialize results list to an empty list
entries = []

# Gets all the combinations "Material"+"Enseña"+"Central Data"+"Familia APO"

# Mode 1 Eroski
if mode == 1:
    cursor.execute(
        'SELECT DISTINCT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZCDATA","_BIC_ZFAMAPO"  FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1"')

# Mode 2 ECI
elif mode == 2:
    cursor.execute(
        'SELECT DISTINCT "_BIC_ZMATERIAL","_BIC_ZENSENA","_BIC_ZCDATA","_BIC_ZFAMAPO"  FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPT01"')

# Mode 3 CRF
elif mode == 3:
    cursor.execute(
        'SELECT DISTINCT "_BIC_ZMATERIA2","_BIC_ZENSENA","_BIC_ZCDATA","_BIC_ZFAMAPO"  FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSD_DE03"')

# Store all the combinations into the list
for row in cursor.fetchall():
    entries.append(row)

print(entries)

interp = False

############### This part of the code has to be executed for each "Material"+"Enseña"+"Control Data"+"Familia APO" combination !!!!!!!!

# To test. In production, this is derived from the "Material"+"Enseña"+"Control Data"+"Familia APO" that is processed
cpt = 0
df_total = None

# We read promotion file and make a new dataframe to use the function "join" in order to calculate promos
if user == "D" and mode == 1:
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1802(1QUINCENA).xlsx"
elif user == "M" and mode == 1:
    promo_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1802(1QUINCENA).xlsx"
elif user == "D" and mode == 2:
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_ECI_2015_1710_VersIII.XLSX"
elif user == "M" and mode == 2:
    promo_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\PROMOCIONES_ECI_2015_1710_VersIII.XLSX"
elif user == "D" and mode == 3:
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_CRF_LYB_DDLL_2015_1709.XLSX"
elif user == "M" and mode == 3:
    promo_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\PROMOCIONES_CRF_LYB_DDLL_2015_1709.XLSX"
elif user == "S" and mode == 1:
    promo_file = "C:\\Datos analisis\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1802(1QUINCENA).xlsx"
elif user == "S" and mode == 2:
    promo_file = "C:\\Datos analisis\\PROMOCIONES_ECI_2015_1710_VersIII.XLSX"
elif user == "S" and mode == 3:
    promo_file = "C:\\Datos analisis\\PROMOCIONES_CRF_LYB_DDLL_2015_1709.XLSX"


promo = pd.read_excel(promo_file)
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
    print(row_promo[1])
    data_matrix.append([row_promo[2], row_promo[7], first_date,
                        row_promo[9], row_promo[10], row_promo[11],
                        row_promo[14], row_promo[8], int(row_promo[1]), row_promo[0]])
    # df_promo.values[cont]=[row_promo[2], row_promo[7], first_date,
    #                    row_promo[9],row_promo[10],row_promo[11],
    #                    row_promo[14],row_promo[9], int(row_promo[1])]
    cont += 1
    for j in range(1, diff.days):
        # print(j)
        # print(row_promo[3]+j+1)
        # print(type(row_promo[3]))
        d = timedelta(days=j)
        data_matrix.append([row_promo[2], row_promo[7], first_date + d,
                            row_promo[9], row_promo[10], row_promo[11],
                            row_promo[14], row_promo[8], int(row_promo[1]), row_promo[0]])
        #   df_promo.loc[cont] = [row_promo[2], row_promo[7], first_date+d,
        #                         row_promo[9], row_promo[10], row_promo[11],
        #                         row_promo[14], row_promo[9], int(row_promo[1])]
        cont += 1
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

# We read the seasonality file
if user == "D":
    station_file = "C:\\Users\\tr5568\\Desktop\\DAYANA\\CAPSA\\Tabla_estacionalidad fichero carga.xlsx"
elif user == "M":
    station_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\Tabla_estacionalidad fichero carga.xlsx"
elif user == "S":
    station_file = "C:\\Datos analisis\\Tabla_estacionalidad fichero carga.xlsx"

station = pd.read_excel(station_file, 1)
for ent in entries:
    #if ent[3] in ["122"]:
    # if ent[1]=="Z5E99K":
    #if ent[3]=="122" and ent[1]=="Z5E99K" and ent[0]=="000000000000011467" and ent[2]=="0000121062":
    # if ent[3] =="550" and ent[1] == "Z5E99K" and ent[0]=="000000000000014129" and ent[2]=="0000121062":
        print("VALOR DE SFAPO: ")
        print(str(ent[3]))
        if (str(ent[3]) == ''):
            SFAPO = 0
        else:
            SFAPO = int(ent[3])
        if mode == 1:
            #### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## This part of the query stays fixed
            # query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
            query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZCDATA","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
            query = query + '"ZFECHA" >= 20160101 AND "_BIC_ZENSENA2" NOT IN (\'Z5E005\',\'Z5E008\',\'Z5E013\',\'Z5E018\') AND '
            ## Here goes the adaptation: replace the hard coded values with the variable of the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## These values are hard coded to test
            query = query + '"_BIC_ZENSENA2" = \'' + ent[1] + '\' AND "_BIC_ZFAMAPO"=\'' + ent[
                3] + '\' AND "_BIC_ZMATERIAL"=\'' + ent[0] + '\' AND "_BIC_ZCDATA"=\'' + ent[2] + '\' '
            ## This part of the query stays fixed
            # query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZDESMER70","ZFECHA" '
            query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZCDATA","ZFECHA" '
            query = query + 'ORDER BY "ZFECHA"'
        elif mode == 2:
            #### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## This part of the query stays fixed
            # query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
            query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA" AS "_BIC_ZENSENA2","_BIC_ZCDATA","_BIC_ZFAMAPO","DATE_SAP_2" AS ZFECHA,sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL", sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPT01" WHERE '
            query = query + '"DATE_SAP_2" >= 20160101 AND "_BIC_ZENSENA" NOT IN (\'Z5E005\',\'Z5E008\',\'Z5E013\',\'Z5E018\') AND '
            ## Here goes the adaptation: replace the hard coded values with the variable of the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## These values are hard coded to test
            query = query + '"_BIC_ZENSENA" = \'' + ent[1] + '\' AND "_BIC_ZFAMAPO"=\'' + ent[
                3] + '\' AND "_BIC_ZMATERIAL"=\'' + ent[0] + '\' AND "_BIC_ZCDATA"=\'' + ent[2] + '\' '
            ## This part of the query stays fixed
            # query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZDESMER70","ZFECHA" '
            query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA","_BIC_ZCDATA","DATE_SAP_2" '
            query = query + 'ORDER BY "DATE_SAP_2"'
        elif mode == 3:
            #### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## This part of the query stays fixed
            # query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
            query = 'SELECT "_BIC_ZCDATA","_BIC_ZMARCA2","_BIC_ZMATERIA2","_BIC_ZSECCION2","_BIC_ZSUBSEC2", "CENTRALDATAT","MARCAT","MATERIALT","SECCIONT","SUBSECCIONT","_BIC_ZDESTMER","DESTINATARIOT1", "_BIC_ZENSENA","_BIC_ZCODPOST","DESTINATARIOT2", "CALMONTH","CALWEEK","ZMM","ZAAAA",sum("_BIC_ZCOUNTER") AS "_BIC_ZCOUNTER",sum("_BIC_ZIMPPVP") AS "_BIC_ZIMPPVP",sum("_BIC_ZUNIDFRA") AS "_BIC_ZUNIDFRA",sum("_BIC_ZVMKLESTA") as "_BIC_ZVMKLESTA" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSD_DE03" GROUP BY "_BIC_ZCDATA","_BIC_ZMARCA2", "_BIC_ZMATERIA2","_BIC_ZSECCION2","_BIC_ZSUBSEC2","CENTRALDATAT","MARCAT","MATERIALT","SECCIONT","SUBSECCIONT","_BIC_ZDESTMER","DESTINATARIOT1", "_BIC_ZENSENA","_BIC_ZCODPOST","DESTINATARIOT2","CALMONTH","CALWEEK", "ZMM","ZAAAA"'
            query = query + '"ZFECHA" >= 20170101 AND "_BIC_ZENSENA2" NOT IN (\'Z5E005\',\'Z5E008\',\'Z5E013\',\'Z5E018\') AND '
            ## Here goes the adaptation: replace the hard coded values with the variable of the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
            ## These values are hard coded to test
            query = query + '"_BIC_ZENSENA2" = \'' + ent[1] + '\' AND "_BIC_ZFAMAPO"=\'' + ent[
                3] + '\' AND "_BIC_ZMATERIAL"=\'' + ent[0] + '\' AND "_BIC_ZCDATA"=\'' + ent[2] + '\' '
            ## This part of the query stays fixed
            # query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZDESMER70","ZFECHA" '
            query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZFAMAPO","_BIC_ZENSENA2","_BIC_ZCDATA","ZFECHA" '
            query = query + 'ORDER BY "ZFECHA"'

        print("SQL Query: " + query)

        # We run the query to get all the entries for a particular "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
        cursor.execute(query)

        # We initialize the result list to an empty list
        rows_list = []

        # We iterate in each row of the result
        for row in cursor.fetchall():
            # print(row)
            # We get the date of the row and cast it to a datetime
            myDate = datetime.datetime.strptime(row[4], '%Y%m%d')

            # If this is not the first row we read
            if (len(rows_list) > 0):
                # We store the date of the last row we read before this one: rows_list[-1] returns the last element of the list
                lastDate = rows_list[-1]["DATE"]
                # We store the difference between the two dates
                diff = (myDate - lastDate).days

                tmp_rows = []

                # We iterate on the number of days there are between the two dates: we want to interpolate between those two dates
                # If we enter this loop, it means that diff > 1 and we have to interpolated the missing days
                for i in range(1, diff):
                    # print("interpolating")
                    # We set the current date to the last date plus i days
                    current_date = lastDate + datetime.timedelta(days=i)
                    # print(current_date)
                    # We initialise the new row we will add to an empty dictionnary
                    dicttmp = {}
                    if interp:
                        # We set the values of the new row to the same values as the last row, except for "CANT", "KL" and "IMP" where we interpolate: we divide the quantity of the last date between the number of missing days. We then set the date for this row to the current date
                        dicttmp.update(
                            {"MAT": rows_list[-1]["MAT"], "ENS": rows_list[-1]["ENS"], "CDATA": int(rows_list[-1]["CDATA"]),
                             "FAMAPO": rows_list[-1]["FAMAPO"], "CANT": rows_list[-1]["CANT"] / float(diff),
                             "KL": rows_list[-1]["KL"] / float(diff),
                             "IMP": rows_list[-1]["IMP"] / float(diff), "DATE": current_date})
                    else:
                        dicttmp.update({"MAT": rows_list[-1]["MAT"], "ENS": rows_list[-1]["ENS"],
                                        "CDATA": int(rows_list[-1]["CDATA"]), "FAMAPO": rows_list[-1]["FAMAPO"],
                                        "CANT": 0.0,
                                        "KL": 0.0,
                                        "IMP": 0.0, "DATE": current_date})
                        # We add the new row to the list of rows we will add to the result
                    tmp_rows.append(dicttmp)
                    # print(dicttmp)
                # print(tmp_rows)
                # If we add to interpolate (diff > 1), we update the last row with the interpolated quantities, as this row is part of the interpolation
                if diff > 1 and interp:
                    rows_list[-1]["CANT"] = rows_list[-1]["CANT"] / float(diff)
                    rows_list[-1]["KL"] = rows_list[-1]["KL"] / float(diff)
                    rows_list[-1]["IMP"] = rows_list[-1]["IMP"] / float(diff)
                rows_list.extend(tmp_rows)
            # print(myDate)
            # Normal row adding phase
            dict1 = {}
            # get input row in dictionary format
            # key = col_name
            if str(row[3]) == '':
                FAMAPO = 0
            else:
                FAMAPO = int(row[3])
            if str(row[2]) == '':
                CDATA = 0
            else:
                CDATA = int(row[2])
            dict1.update(
                {"MAT": row[0], "ENS": row[1], "CDATA": CDATA, "FAMAPO": FAMAPO, "CANT": float(row[5]), "KL": float(row[6]),
                 "IMP": float(row[7]), "DATE": myDate})
            rows_list.append(dict1)

        # We build the complete dataframe with all the rows: now we have exactly one row per day, without any missing value
        print("CONSTRUYENDO TOTAL DF")
        if (len(rows_list) > 0):
            total = pd.DataFrame(rows_list)

            # We add a column with the week number
            total["WEEK"] = total.apply(func, axis=1)
            # print(total)

            # See if we have to detrend: we look if the SFAPO that we are computing is present in the seasonality file
            vector_station = station.loc[:, "cod sfapo"].values
            print(SFAPO in station.loc[:, "cod sfapo"])
            if SFAPO in vector_station:
                print("Detrending")
                # print(station[station["cod sfapo"]==SFAPO])
                total = pd.merge(total, station, left_on=["FAMAPO", "WEEK"], right_on=["cod sfapo", "semana"])
                total["TREND"] = total.apply(calc_trend, axis=1)
                total = total.drop("cod sfapo", 1)
                total = total.drop("semana", 1)
                total = total.drop("volumen", 1)
            else:
                print("NOT Detrending")
                total["TREND"] = 1.0

            total = total.sort_values(by=['DATE'])

            # Now we have a dataframe with a trend column
            total["KL_DETREND"] = total.loc[:, "KL"] * total.loc[:, "TREND"]
            total["EUROS_DETREND"] = total.loc[:, "IMP"] * total.loc[:, "TREND"]
            # Now we have a dataframe with detrended columns
            # print("CHECK 1 BIS OF TOTAL")
            # print(total)
            # print(total)
            # if cpt ==0:
            #    total.to_csv("Dayana.csv", sep=",", index = False)
            # else:
            #    total.to_csv("Dayana.csv", mode='a', header=False, sep=",", index=False)

            # insert promo columns in dataframe using join
            total = total.join(
                df_promo.set_index(['FAMAPO', 'DATE', 'ENS', 'CDATA']),
                on=['FAMAPO', 'DATE', 'ENS', 'CDATA'])
            # print("ANIMACIÓN 1")
            # print(total["Animacion 1"])

            total.replace({'Animacion 1': {None: 0}}, inplace=True)
            total.replace({'Animacion 2': {None: 0}}, inplace=True)
            total.replace({'Animacion 3': {None: 0}}, inplace=True)
            total.replace({'TEMATICA': {None: 0}}, inplace=True)
            total.replace({'Abreviatura accion': {None: 0}}, inplace=True)
            total.replace({'Codigo unico': {None: 0}}, inplace=True)

            # we calculate a new row called STATUS PROMO ("P" if there is promo)
            total["STATUS_PROMO"] = total.apply(ispromo, axis=1)

            # print("STATUS PROMO")
            # print(total["STATUS_PROMO"])
            # reset_index
            total = total.reset_index(drop=True)

            # BASELINE calculation
            BASELINE = np.array(total.loc[:, "KL_DETREND"].copy())
            BASELINE2 = BASELINE.copy()
            old_baseline = []
            means = []
            # using windows
            wS = 5
            hWS = wS // 2
            print(total)
            bs2WS = 60
            bs2hWS = bs2WS // 2
            if len(BASELINE) >= wS:
                for i, x in enumerate(BASELINE):
                    dayOfWeek = (total.values[i, 2]).isoweekday()
                    # print(dayOfWeek)
                    days = []
                    # Baseline2
                    # print("BASELINE2")
                    if i >= bs2hWS and i < len(BASELINE) - bs2hWS:
                        vector = total.values[i - bs2hWS:i + bs2hWS + 1, :]
                        # print(vector)


                    elif i in range(0, bs2hWS):
                        vector = total.values[0:bs2WS, :]
                        # print("VECTOR")
                        # print(vector)
                    elif i in range(len(BASELINE) - bs2hWS, len(BASELINE)):
                        vector = total.values[len(BASELINE) - (bs2WS + 1):len(BASELINE), :]

                    # Compute means for each day of the week
                    # print("LENGTH VECTOR")
                    # print(len(vector))
                    # print(vector)
                    for it in range(0, len(vector)):
                        if vector[it, 18] != "P" and (
                                vector[it, 2]).isoweekday() == dayOfWeek:
                            days.append(vector[it, 10])
                    if len(days) == 0:
                        print("DID NOT FIND ANY DAY FOR " + str(total["DATE"].iloc[i]) + " !!!!")
                        # print(vector)
                    # print(days)
                    meanDay = np.mean(days)
                    # print(meanDay)
                    BASELINE2[i] = meanDay

                    # normal baseline
                    if mode_baseline == 1:
                        min = 999999999999999999.99
                        total_average = 0
                        contador = 0
                        average = 0
                        if i >= hWS and i < len(BASELINE) - hWS:
                            vector = BASELINE[i - hWS:i + hWS + 1]
                        elif i in range(0, hWS):
                            vector = BASELINE[0:wS]
                        elif i in range(len(BASELINE) - hWS, len(BASELINE)):
                            vector = BASELINE[len(BASELINE) - (wS + 1):len(BASELINE)]

                        no_out = reject_outliers(vector, 1.7)
                        # print(vector)
                        # print(no_out)

                        average = np.mean(no_out)

                        # print("VALOR")
                        # print(x)
                        # print("VECTOR")
                        # print(vector)
                        var = average * 0.50

                        for j, y in enumerate(vector):
                            if y >= average - var:
                                if y < min:
                                    min = y
                        means.append(average)

                # print(days)
                # BASELINE = np.array(means)

                # print(len(means))
                # print("LENGTH")
                # print(len(BASELINE))
                # print(len(BASELINE2))
                # print(BASELINE2)
                # BASELINE = np.array(means)
                # BASELINE = BASELINE2
            # plt.plot(BASELINE)
            # Replace 0 for median of BASELINE vector (without 0 values)
            # median = float(np.median(BASELINE[BASELINE > 0]))
            # BASELINE[BASELINE == 0] = median

            # print(BASELINE)
            # print(len(BASELINE))

            average_KL_DETREND = BASELINE.mean()

            if mode_baseline == 1:
                # we want to replace values of baseline in promo days for average of KL_DETREND in days without promo
                average_KL_DETREND_nopromo = 0

                print("AV KL_DETREND")
                print(average_KL_DETREND)

                #   #KL_DETREND column number=10
                #   if x!="P":
                #       average_KL_DETREND_nopromo+=total.loc[i,"KL_DETREND"]
                #       aux += 1

                # print("AVERAGE")
                # print(average_KL_DETREND_nopromo)
                # plt.plot(BASELINE)
                # plt.plot(total.loc[:,"KL_DETREND"])
                # plt.ylabel('some numbers')
                # plt.show()
                total["BASELINE"] = BASELINE
                for i, x in enumerate(total.values):
                    total_aux = total[(total["DATE"] <= (x[2] + timedelta(days=40))) & (
                                total["DATE"] >= (x[2] - timedelta(days=40)))].reset_index(drop=True)
                    # #     print(str(x[2]))
                    # print("LEN total_aux")
                    # print(len(total_aux))
                    # print(total_aux)
                    aux = 0
                    for j in range(0, len(total_aux)):
                        if total_aux.loc[j, "STATUS_PROMO"] != "P" and total_aux.loc[j, "KL_DETREND"] != 0:
                            average_KL_DETREND_nopromo += total_aux.loc[j, "BASELINE"]
                            aux += 1
                    if aux != 0: average_KL_DETREND_nopromo = average_KL_DETREND_nopromo / aux
                    # #    if x[18]=="P": BASELINE[i]=average_KL_DETREND
                    if x[18] == "P": BASELINE[i] = average_KL_DETREND_nopromo

                # print("BASELINE CON KL_DETREND AVERAGE EN DÍAS CON PROMO")
                # print(BASELINE)

                # print("BASELINE ANTES DE SAV")
                # print(BASELINE)
                # print(len(BASELINE))
                # Savitzky
                print("BASELINE")
                print(type(BASELINE[0]))
                print(BASELINE)
                ventana = 21
                if (len(BASELINE) > 30):
                    if len(BASELINE) < ventana:
                        if len(BASELINE) % 2 != 1:
                            ventana = len(BASELINE) - 1
                        else:
                            ventana = len(BASELINE)
                    BASELINE = savitzky_golay(BASELINE, ventana, 2)  # window size 51, polynomial order 3

                # BASELINE = slicing(BASELINE)
                # BASELINE = peakutils.baseline(BASELINE, deg=6, max_it=1000, tol=0.000001)
                # BASELINE = baseline_als(BASELINE)
                # BASELINE=smooth(BASELINE, ventana, window="blackman")

                # print("BASELINE DESPUÉS DE SAV")
                # print(BASELINE)
                # print(len(BASELINE))


                # print("TAM DE TOTAL")
                # print(len(total))
                # Add BASELINE column to our dataframe
                # Check if total is ordered
                print("CHECK TOTAL")
                print(total)

                for i, x in enumerate(total.values):
                    if x[2].isoweekday() == 6: BASELINE[i] = x[10]

            elif mode_baseline == 2:
                BASELINE = BASELINE2.copy()

            treshold = float(0.20 * average_KL_DETREND)
            for i, x in enumerate(total["KL_DETREND"]):
                if x <= treshold:
                    BASELINE[i] = x

            total["BASELINE"] = BASELINE
            # in days with low values of KL_DETREND we have to replace BASELINE value (BASELINE=KL_DETREND)
            # total["BASELINE"]=total.apply(replace,axis=1)
            # Add incremental KL_DETREND column to our dataframe
            total["VENTA_INCREMENTAL"] = total.loc[:, "KL_DETREND"] - total.loc[:, "BASELINE"]
            # Add VENTA_PROMO to our dataframe
            total["VENTA_PROMO"] = total.apply(ventapromo, axis=1)
            # Add EUROS_PROMO to our dataframe
            total["EUROS_PROMO"] = total.apply(eurospromo, axis=1)
            # print(total)
            # plt.plot(BASELINE)
            # plt.plot(total.loc[:,"KL_DETREND"])
            # plt.ylabel('some numbers')
            # plt.show()
            # print("LONGITUD DE TOTAL")
            # print(len(total))

            if df_total is None:
                df_total = total
                print("None")
            else:
                df_total = df_total.append(total, ignore_index=True)
                print("append")

        cpt += 1
        print("NÚMERO DE QUERYS REALIZADAS")
        print(cpt)

# total["BASELINE"] = BASELINE2

# print("DF ANTES DE CANIB")
# print(df_total)

# We read canib file
if user == "D":
    canib_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\GRUPOS CANIBALIZACIÓN FEB_18.xlsx"
elif user == "M":
    canib_file = "C:\\Users\\gnuma\\Google Drive\\CAPSA\\Softwares\\GRUPOS CANIBALIZACIÓN FEB_18.xlsx"
elif user == "S":
    canib_file = "C:\\Datos analisis\\GRUPOS CANIBALIZACIÓN FEB_18.xlsx"

canib_excel = pd.read_excel(canib_file)
df_total = df_total.join(canib_excel.set_index('Cod. Familia'), on='FAMAPO')
# print(df_total)
df_total.replace({'Grupo canibalizacion': {None: -1}}, inplace=True)

# group by Grupo canibalizacion and DATE
data_canib = df_total.groupby(['Grupo canibalizacion', 'DATE', 'CDATA','ENS'])
print(len(data_canib))
dict_promo = {}
# we create a dictionary which shows if each group of "Grupo canibalizacion_DATE_CDATA_ENS" has promo("P") or not
print(type(data_canib))
for df in data_canib:
    # print(df)
    # print("df[0]")
    # print(df[0])
    # print("df[1]")
    # print(df[1])
    codigo_unico = []
    if df[0] != -1:
        if "P" in df[1].reset_index(drop=True).loc[:, "STATUS_PROMO"].values:
            datafr = df[1].reset_index(drop=True).loc[:, "Codigo unico"]
            codigo_unico = datafr[datafr != 0].reset_index(drop=True)
            # print("CODIGO UNICO")
            # print(codigo_unico)
            dict_promo[str(df[1].reset_index(drop=True).loc[0, "Grupo canibalizacion"]) + "_" + str(
                df[1].reset_index(drop=True).loc[0, "DATE"])+"_"+ str(df[1].reset_index(drop=True).loc[0,"CDATA"])+"_"+
                       str(df[1].reset_index(drop=True).loc[0,"ENS"])]= codigo_unico[0]
            # print("RELLENANDO CON CÓDIGO ÚNICO DE PROMO")
        else:
            dict_promo[str(df[1].reset_index(drop=True).loc[0, "Grupo canibalizacion"]) + "_" + str(
                df[1].reset_index(drop=True).loc[0, "DATE"]) + "_" + str(df[1].reset_index(drop=True).loc[0, "CDATA"]) + "_"
                       + str(df[1].reset_index(drop=True).loc[0, "ENS"])] = 0
            # print("RELLENANDO CON 0")

print("DICCIONARIO PROMOS")
print(dict_promo)
vector = []

# if we have a product without with the same "Grupo canibalizacion" and "DATE" of other in promo
# we have to change "STATUS_PROMO" vector and we put a 'C' instead of "0"

matriz_aux = df_total.values
for i, x in enumerate(matriz_aux):
    # row=df_total.iloc[i,:]
    key = str(x[23]) + "_" + str(x[2])+ "_" + str(x[1]) + "_" + str(x[3])
    # key=str(row["Grupo canibalizacion"])+"_"+str(row["DATE"])

    if key in dict_promo:
        if dict_promo[key] != 0:
            # df_total[i,"STATUS_PROMO"]="C"
            # vector.append("C")
            if x[18] != "P":
                matriz_aux[i, 18] = "C"
                matriz_aux[i, 17] = dict_promo[key]
                print("CODIGO PROMO", dict_promo[key])
                print(i)
                print("CAMBIANDO EL VALOR A C")

for i, x in enumerate(matriz_aux):
    # if no promo, venta_incremental=0
    if x[18] not in ["P", "C"]:  matriz_aux[i, 20] = 0

    # aux_df=df_total[(df_total["Grupo canibalizacion"]==row["Grupo canibalizacion"]) & (df_total["DATE"]==row["DATE"])]
        # print("DATAFRAME AUX")
        # print(aux_df)
        # print("COLUMNA STATUS PROMO")
        # print(aux_df["STATUS_PROMO"])
        # if any(aux_df["STATUS_PROMO"]=="P"):
        # if "P" in aux_df["STATUS_PROMO"].values:
        # df_total[i,"STATUS_PROMO"]="C"
        # row["STATUS_PROMO"]="C"
        # print(i)
        # print("ENTRO EN EL BUCLE Y CAMBIO EL DATO A C")

# we have a new df with sum and size of BASELINE and KL_DETREND
# we need this df to calculate canibalizacion
# canibalizacion =
# mean of KL_DETREND of all the products with the same Grupo canibalizacion (without the one we are considering) -
# -mean of BASELINE of all the products with the same Grupo canibaliacion (without the one we are considering)
# df_baseline=data_canib['BASELINE'].agg(['sum','size']).reset_index()
# df_KLdetrend=data_canib['KL_DETREND'].agg(['sum','size']).reset_index()
# df_baseline.to_csv("df_baseline.csv", sep=',')
# df_KLdetrend.to_csv("df_KLdetrend.csv", sep=',')

# canib=[]

# for i in range(0, len(df_total.index)):
#    row_data = df_total.iloc[i]
# print(row_data)
#    if row_data["KL_DETREND"] == 0 or row_data["Grupo canibalizacion"] == -1:
#        canib.append(0)
#    else:
#        aux_base = df_baseline[(df_baseline['Grupo canibalizacion'] == row_data['Grupo canibalizacion']) & (df_baseline['DATE'] == row_data["DATE"])]
#        aux_KLdetrend = df_KLdetrend[(df_KLdetrend['Grupo canibalizacion'] == row_data["Grupo canibalizacion"]) & (df_KLdetrend['DATE'] == row_data["DATE"])]
# print((aux_base['sum']-row_data[13])/(aux_base['size']-1)-((aux_KLdetrend['sum']-row_data[11])/(aux_KLdetrend['size']-1)))
# print('aux baseline')
# print(aux_base)
# print('aux KL detrend')
# print(aux_KLdetrend)
#        if (aux_KLdetrend['size'].iloc[0] <= 1):
#            canib.append(0)
#        else:
#            canib.append(float(((aux_KLdetrend['sum'] - row_data["KL_DETREND"]) / (aux_KLdetrend['size'] - 1)) - (
#            (aux_base['sum'] - row_data["BASELINE"]) / (aux_base['size'] - 1))))

# add the new calculated column to our data
# df_total["CANIBALIZACION"]=canib

# df_total.index=len(df_total)
# print("DF CON CANIBALIZACIÓN")
# print(df_total)
# we write data in a csv file
print("Writing final CSV file")
df_total2 = pd.DataFrame(matriz_aux,
                         columns=["CANT", "CDATA", "DATE", "ENS", "FAMAPO", "IMP", "KL", "MAT", "WEEK", "TREND",
                                  "KL_DETREND", "EUROS_DETREND", "Animacion 1", "Animacion 2", "Animacion 3",
                                  "TEMATICA", "Abreviatura accion", "Codigo unico", "STATUS_PROMO", "BASELINE",
                                  "VENTA_INCREMENTAL",
                                  "VENTA_PROMO", "EUROS_PROMO", "Grupo canibalizacion"])
df_total2["TREND"] = df_total2["TREND"].astype(float)
df_total2["KL_DETREND"] = df_total2["KL_DETREND"].astype(float)
df_total2["EUROS_DETREND"] = df_total2["EUROS_DETREND"].astype(float)
df_total2["BASELINE"] = df_total2["BASELINE"].astype(float)
df_total2["VENTA_INCREMENTAL"] = df_total2["VENTA_INCREMENTAL"].astype(float)
df_total2["VENTA_PROMO"] = df_total2["VENTA_PROMO"].astype(float)
df_total2["EUROS_PROMO"] = df_total2["EUROS_PROMO"].astype(float)
df_total2["CANT"] = df_total2["CANT"].astype(float)
df_total2["IMP"] = df_total2["IMP"].astype(float)
df_total2["KL"] = df_total2["KL"].astype(float)
# df_total2["MEANS"] = BASELINE2

# print(df_total2["MEANS"])
df_total2.to_csv("data_Eroski_2018_02_23.csv", sep=';', decimal=',', float_format='%.6f')
print("Finished writing file")
