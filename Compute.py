import pyodbc
import pandas as pd
import datetime
import numpy as np
from math import factorial
import statistics as stats
import matplotlib.pyplot as plt

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
    if row["TEMAT"]==0:
        return 0
    else: return row["KL_DETREND"]

#Takes a row and returns EUROS_DETREND or 0 corresponding to the value in promo columns
def eurospromo(row):
    if row["TEMAT"]=="0":
        return 0
    else: return row["EUROS_DETREND"]

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
print(len(entries))

interp = False

############### This part of the code has to be executed for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination !!!!!!!!

# To test. In production, this is derived from the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" that is processed
cpt = 0
df_total=None
# We read promotion file
promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1710_II.xlsx"
promo = pd.read_excel(promo_file)
# We read the seasonality file
station_file = "C:\\Users\\tr5568\\Desktop\\DAYANA\\CAPSA\\Tabla_estacionalidad fichero carga.xlsx"
station = pd.read_excel(station_file, 1)
for ent in entries:
    if ent[3] in ["340", "341","360", "366","470","471"] and ent[1] == "Z5E99K":
    #if ent[3] =="341" and ent[1] == "Z5E99K" and ent[0]=="000000000000013225" and ent[2]=="0000005317":
        SFAPO = int(ent[3])

        #### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
        ## This part of the query stays fixed
        #query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
        query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZCDATA","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
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

        # We initialize the result list to an empty list
        rows_list = []

        # We iterate in each row of the result
        for row in cursor.fetchall():
            # We get the date of the row and cast it to a datetime
            myDate = datetime.datetime.strptime(row[4],'%Y%m%d')

            # If this is not the first row we read
            if(len(rows_list)>0):
                # We store the date of the last row we read before this one: rows_list[-1] returns the last element of the list
                lastDate = rows_list[-1]["DATE"]
                # We store the difference between the two dates
                diff = (myDate - lastDate).days

                tmp_rows = []


                # We iterate on the number of days there are between the two dates: we want to interpolate between those two dates
                # If we enter this loop, it means that diff > 1 and we have to interpolated the missing days
                for i in range(1,diff):
                    #print("interpolating")
                    # We set the current date to the last date plus i days
                    current_date = lastDate + datetime.timedelta(days=i)
                    #print(current_date)
                    # We initialise the new row we will add to an empty dictionnary
                    dicttmp = {}
                    if interp:
                        # We set the values of the new row to the same values as the last row, except for "CANT", "KL" and "IMP" where we interpolate: we divide the quantity of the last date between the number of missing days. We then set the date for this row to the current date
                        dicttmp.update({"MAT": rows_list[-1]["MAT"], "ENS":rows_list[-1]["ENS"], "CDATA": rows_list[-1]["CDATA"], "FAMAPO": rows_list[-1]["FAMAPO"], "CANT": rows_list[-1]["CANT"]/float(diff), "KL": rows_list[-1]["KL"]/float(diff),
                                     "IMP": rows_list[-1]["IMP"]/float(diff), "DATE": current_date})
                    else:
                        dicttmp.update({"MAT": rows_list[-1]["MAT"], "ENS": rows_list[-1]["ENS"],
                                            "CDATA": rows_list[-1]["CDATA"], "FAMAPO": rows_list[-1]["FAMAPO"],
                                            "CANT": 0.0,
                                            "KL": 0.0,
                                            "IMP": 0.0, "DATE": current_date})
                        # We add the new row to the list of rows we will add to the result
                    tmp_rows.append(dicttmp)
                    #print(dicttmp)
                #print(tmp_rows)
                # If we add to interpolate (diff > 1), we update the last row with the interpolated quantities, as this row is part of the interpolation
                if diff > 1 and interp:
                    rows_list[-1]["CANT"] = rows_list[-1]["CANT"]/float(diff)
                    rows_list[-1]["KL"] = rows_list[-1]["KL"] / float(diff)
                    rows_list[-1]["IMP"] = rows_list[-1]["IMP"] / float(diff)
                rows_list.extend(tmp_rows)
            #print(myDate)
            # Normal row adding phase
            dict1 = {}
            # get input row in dictionary format
            # key = col_name
            dict1.update({"MAT":row[0], "ENS":row[1], "CDATA":row[2], "FAMAPO":int(row[3]), "CANT":float(row[5]), "KL":float(row[6]), "IMP":float(row[7]), "DATE":myDate})
            #print(dict1)



            value=False
            #we search if the product in dict has promo
            print("ANTES DE BUCLE FOR EN PROMOS")
            for j in range(0, len(promo.index)):
                row_promo = promo.loc[j, :]
                if (not np.math.isnan(row_promo[7])):
                    if dict1["ENS"]==row_promo[2] and dict1["FAMAPO"]==row_promo[7]:
                        if dict1["DATE"]<=row_promo[4] and dict1["DATE"]>=row_promo[3]:
                            value=True
                            row_aux=j

            print("DESPUÉS DE BUCLE FOR EN PROMOS")
            #value=True --> the product has promo
            if value:
                dict1.update({"ANIM1":promo.iloc[row_aux]["Animacion 1"], "ANIM2":promo.iloc[row_aux]["Animacion 2"],
                              "ANIM3":promo.iloc[row_aux]["Animacion 3"],
                              "ABREVACC":promo.iloc[row_aux]["Abreviatura accion"],
                              "TEMAT":promo.iloc[row_aux]["TEMATICA"]})

            else:
                dict1.update({"ANIM1": 0, "ANIM2": 0,
                              "ANIM3": 0, "ABREVACC": 0,"TEMAT": 0})

            #print(dict1)
            rows_list.append(dict1)

        # We build the complete dataframe with all the rows: now we have exactly one row per day, without any missing value
        print("CONSTRUYENDO TOTAL DF")
        if(len(rows_list)>0):
            total = pd.DataFrame(rows_list)
            #print(total)

            # We add a column with the week number
            total["WEEK"] = total.apply(func, axis=1)
            #print(total)


            # See if we have to detrend: we look if the SFAPO that we are computing is present in the seasonality file
            if SFAPO in station.loc[:,"cod sfapo"]:
                print("Detrending")
                #print(station[station["cod sfapo"]==SFAPO])
                total = pd.merge(total, station, left_on=["FAMAPO", "WEEK"], right_on=["cod sfapo", "semana"])
                total["TREND"] = total.apply(calc_trend, axis = 1)
                total = total.drop("cod sfapo",1)
                total = total.drop("semana", 1)
                total = total.drop("volumen", 1)
            else:
                print("NOT Detrending")
                total["TREND"] = 1.0

            # Now we have a dataframe with a trend column
            total["KL_DETREND"] = total.loc[:, "KL"] * total.loc[:, "TREND"]
            total["EUROS_DETREND"] = total.loc[:, "IMP"] * total.loc[:, "TREND"]
            # Now we have a dataframe with detrended columns

            #print(total)
            #if cpt ==0:
            #    total.to_csv("Dayana.csv", sep=",", index = False)
            #else:
            #    total.to_csv("Dayana.csv", mode='a', header=False, sep=",", index=False)


            #BASELINE calculation
            BASELINE=np.array(total.loc[:,"KL_DETREND"])

            #using windows
            if len(BASELINE) >= 7:

                for i, x in enumerate(BASELINE):
                    if i >= 3 and i < len(BASELINE) - 3:
                        average = 0
                        vector = []
                        vector.append(BASELINE[i])
                        for j in range(1, 4):
                            vector.append(BASELINE[i - 1])
                            vector.append(BASELINE[i + 1])
                        for j, y in enumerate(vector):
                            average += y
                        average = average / len(vector)
                        var = stats.stdev(vector, average)
                        #vector_average.append(average)
                        #vector_var.append(var)
                        if abs(BASELINE[i]) > average + var or abs(BASELINE[i]) < average - var:
                            BASELINE[i] = average
                    else:
                        if i in [0, 1, 2]:
                            vector = []
                            average = 0
                            for b in range(0, 7):
                                vector.append(BASELINE[i])
                            for b, x in enumerate(vector):
                                average += x
                            average = average / len(vector)
                            var = stats.stdev(vector, average)
                            #vector_average.append(average)
                            #vector_var.append(var)
                            if abs(BASELINE[i]) > average + var or abs(BASELINE[i]) < average - var:
                                BASELINE[i] = average

                        if i in [len(BASELINE) - 1, len(BASELINE) - 2, len(BASELINE) - 3]:
                            vector = []
                            average = 0
                            for b in range(1, 8):
                                vector.append(BASELINE[len(BASELINE) - b])
                            for j, y in enumerate(vector):
                                average += y
                            average = average / len(vector)
                            var = stats.stdev(vector, average)
                            #vector_average.append(average)
                            #vector_var.append(var)
                            if abs(BASELINE[i]) > average + var or abs(BASELINE[i]) < average - var:
                                BASELINE[i] = average


            #plt.plot(BASELINE)
            #Replace 0 for median of BASELINE vector (without 0 values)
            median=float(np.median(BASELINE[BASELINE>0]))
            BASELINE[BASELINE==0]=median

            #Savitzky
            ventana=51
            if(len(BASELINE)>30):
                if len(BASELINE)<ventana:
                    if len(BASELINE)%2!=1:
                        ventana=len(BASELINE)-1
                    else:
                        ventana=len(BASELINE)
                BASELINE= savitzky_golay(BASELINE, ventana, 2)  # window size 51, polynomial order 3


            #Add BASELINE column to our dataframe
            total["BASELINE"]=BASELINE
            #Add incremental KL_DETREND column to our dataframe
            total["VENTA_INCREMENTAL"] = total.loc[:, "KL_DETREND"] - total.loc[:, "BASELINE"]
            #Add VENTA_PROMO to our dataframe
            total["VENTA_PROMO"]=total.apply(ventapromo, axis=1)
            #Add EUROS_PROMO to our dataframe
            total["EUROS_PROMO"] = total.apply(eurospromo, axis=1)
            #print(total)
            #plt.plot(BASELINE)
            #plt.plot(total.loc[:,"KL_DETREND"])
            #plt.ylabel('some numbers')
            #plt.show()


            if df_total is None:
                df_total=total
            else:
                df_total.append(total)
        cpt += 1




#We read canib file
canib_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\Canib.xlsx"
canib_excel = pd.read_excel(canib_file)
df_total = df_total.join(canib_excel.set_index('Cod. Familia'), on='FAMAPO')
print(df_total)
df_total.replace({'Grupo canibalizacion': {None: -1}}, inplace=True)
#group by Grupo canibalizacion and DATE
print(df_total)
data_canib=df_total.groupby(['Grupo canibalizacion', 'DATE'])


#we have a new df with sum and size of BASELINE and KL_DETREND
#we need this df to calculate canibalizacion
#canibalizacion =
#mean of KL_DETREND of all the products with the same Grupo canibalizacion (without the one we are considering) -
# -mean of BASELINE of all the products with the same Grupo canibaliacion (without the one we are considering)
df_baseline=data_canib['BASELINE'].agg(['sum','size']).reset_index()
df_KLdetrend=data_canib['KL_DETREND'].agg(['sum','size']).reset_index()
#df_baseline.to_csv("df_baseline.csv", sep=',')
#df_KLdetrend.to_csv("df_KLdetrend.csv", sep=',')

canib=[]

for i in range(0, len(df_total.index)):
    row_data = df_total.loc[i, :]
    # print(row_data)
    if row_data[10] == 0 or row_data[14] == -1:
        canib.append(0)
    else:
        aux_base = df_baseline[(df_baseline['Grupo canibalizacion'] == row_data[21]) & (df_baseline['DATE'] == row_data[5])]
        aux_KLdetrend = df_KLdetrend[(df_KLdetrend['Grupo canibalizacion'] == row_data[21]) & (df_KLdetrend['DATE'] == row_data[5])]
        # print((aux_base['sum']-row_data[13])/(aux_base['size']-1)-((aux_KLdetrend['sum']-row_data[11])/(aux_KLdetrend['size']-1)))
        print('aux baseline')
        print(aux_base)
        print('aux KL detrend')
        print(aux_KLdetrend)
        if (aux_KLdetrend['size'].iloc[0] <= 1):
            canib.append(0)
        else:
            canib.append(float(((aux_KLdetrend['sum'] - row_data[15]) / (aux_KLdetrend['size'] - 1)) - (
            (aux_base['sum'] - row_data[17]) / (aux_base['size'] - 1))))

#add the new calculated column to our data
df_total["CANIBALIZACION"]=canib

# we write data in a csv file
df_total.to_csv("data_canib.csv", sep=',')