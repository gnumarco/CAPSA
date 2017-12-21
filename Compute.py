import pyodbc
import pandas as pd
import datetime

# Takes a row and returns the week number corresponding to the date in the DATE column
def func(row):
    return(row["DATE"].isocalendar()[1])

# Takes a row and returns the trend value corresponding to the value in the cilumn "volumen"
def calc_trend(row):
    return(1.0/( row["volumen"]/(1.0/52.0)))

# Establish connection to SAP HANA server
cnxn = pyodbc.connect('Driver=HDBODBC;SERVERNODE=bwdbprod:30041;UID=SAPEP01;PWD=EfiProm2017')

# Gets a cursor on the server to perform query
cursor = cnxn.cursor()

# Initialize results list to an empty list
entries = []

# Gets all the combinations "Material"+"Enseña"+"Punto de Venta"+"Familia APO"
cursor.execute(
    'SELECT DISTINCT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO"  FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1"')

# Store all the combinations into the list
for row in cursor.fetchall():
    entries.append(row)

print(entries)

############### This part of the code has to be executed for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination !!!!!!!!

# To test. In production, this is derived from the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" that is processed
SFAPO = 111

#### This query has to be adapted for each "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
## This part of the query stays fixed
query = 'SELECT "_BIC_ZMATERIAL","_BIC_ZENSENA2","_BIC_ZDESMER70","_BIC_ZFAMAPO","ZFECHA",sum("_BIC_ZCANTOT") AS "_BIC_ZCANTOT",sum("_BIC_ZKL") AS "_BIC_ZKL",sum("_BIC_ZIMPTOT2") AS "_BIC_ZIMPTOT2","_BIC_ZSECTOR","_BIC_ZCONSERVA","_BIC_ZTIPMAT","_BIC_ZSECCION","_BIC_ZSUBSEC","_BIC_ZFAMILIA","_BIC_ZSUBFAMIL","_BIC_ZVARIEDAD","_BIC_ZFORMATO","MATL_GROUP","BASE_UOM","_BIC_ZMARCA","_BIC_ZTIPOIVA","_BIC_ZENVASE","EANUPC","NET_WGT_DL","GROSS_WT","UNIT_OF_WT","_BIC_ZPMATN","_BIC_ZEIVR","TXTLG","_BIC_ZENSENA","_BIC_ZDESMER69","_BIC_ZEMISOR","ZDESMER69_TXTMD","ZEMISOR_TXTMD","ZEMISOR__BIC_ZEMISOR","ZENSENA__BIC_ZENSENA","TXTSH","ZYEAR","ZMONTH" FROM "_SYS_BIC"."CAPSA_BW_01.ZEP1/ZSLSRPTF1" WHERE '
query = query + '"ZFECHA" >= 20170101 AND "_BIC_ZENSENA2" NOT IN (\'Z5E005\',\'Z5E008\',\'Z5E013\',\'Z5E018\') AND '
## Here goes the adaptation: replace the hard coded values with the variable of the "Material"+"Enseña"+"Punto de Venta"+"Familia APO" combination
## These values are hard coded to test
query = query + '"_BIC_ZENSENA2" = \'Z5E999\' AND "_BIC_ZFAMAPO"=\'111\' AND "_BIC_ZMATERIAL"=\'000000000000011070\' AND "_BIC_ZDESMER70"=\'0000007009\' '
## This part of the query stays fixed
query = query + 'GROUP BY "_BIC_ZMATERIAL","_BIC_ZSECTOR","_BIC_ZCONSERVA","_BIC_ZTIPMAT","_BIC_ZSECCION","_BIC_ZSUBSEC","_BIC_ZFAMILIA","_BIC_ZSUBFAMIL","_BIC_ZVARIEDAD","_BIC_ZFORMATO","MATL_GROUP","BASE_UOM","_BIC_ZMARCA","_BIC_ZTIPOIVA","_BIC_ZENVASE","EANUPC","_BIC_ZFAMAPO","NET_WGT_DL","GROSS_WT","UNIT_OF_WT","_BIC_ZPMATN","_BIC_ZEIVR","TXTLG","_BIC_ZENSENA","_BIC_ZDESMER69","_BIC_ZEMISOR","_BIC_ZENSENA2","_BIC_ZDESMER70","ZDESMER69_TXTMD","ZEMISOR_TXTMD","ZEMISOR__BIC_ZEMISOR","ZENSENA__BIC_ZENSENA","TXTSH","ZFECHA","ZYEAR","ZMONTH" '
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
            # We set the values of the new row to the same values as the last row, except for "CANT", "KL" and "IMP" where we interpolate: we divide the quantity of the last date between the number of missing days. We then set the date for this row to the current date
            dicttmp.update({"MAT": rows_list[-1]["MAT"], "ENS":rows_list[-1]["ENS"], "PTVENTA": rows_list[-1]["PTVENTA"], "FAMAPO": rows_list[-1]["FAMAPO"], "CANT": rows_list[-1]["CANT"]/float(diff), "KL": rows_list[-1]["KL"]/float(diff),
                     "IMP": rows_list[-1]["IMP"]/float(diff), "DATE": current_date})
            # We add the new row to the list of rows we will add to the result
            tmp_rows.append(dicttmp)
            #print(dicttmp)
        #print(tmp_rows)
        # If we add to interpolate (diff > 1), we update the last row with the interpolated quantities, as this row is part of the interpolation
        if diff > 1:
            rows_list[-1]["CANT"] = rows_list[-1]["CANT"]/float(diff)
            rows_list[-1]["KL"] = rows_list[-1]["KL"] / float(diff)
            rows_list[-1]["IMP"] = rows_list[-1]["IMP"] / float(diff)
        rows_list.extend(tmp_rows)
    #print(myDate)
    # Normal row adding phase
    dict1 = {}
    # get input row in dictionary format
    # key = col_name
    dict1.update({"MAT":row[0], "ENS":row[1], "PTVENTA":row[2], "FAMAPO":int(row[3]), "CANT":float(row[5]), "KL":float(row[6]), "IMP":float(row[7]), "DATE":myDate})
    #print(dict1)
    rows_list.append(dict1)

# We build the complete dataframe with all the rows: now we have exactly one row per day, without any missing value
total = pd.DataFrame(rows_list)
print(total)

# We add a column with the week number
total["WEEK"] = total.apply(func, axis=1)
print(total)

# We read the seasonality file
station_file = "C:\\Users\\Marc\\Google Drive\\CAPSA\\New Data\\Tabla_estacionalidad fichero carga.xlsx"
station = pd.read_excel(station_file,1)
#print(station)

# See if we have to detrend: we look if the SFAPO that we are computing is present in the seasonality file
if SFAPO in station.loc[:,"cod sfapo"]:
    print("Detrending")
    print(station[station["cod sfapo"]==SFAPO])
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
print(total)
