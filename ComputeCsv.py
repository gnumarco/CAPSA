import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import glob
import datetime
import xlrd

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

#we read each file in the path and make calculations
path=r'C:/Users/tr5568/Desktop/DAYANA/CAPSA/csv'
allFiles=glob.glob(path+"/*.csv")
print(allFiles)
#data=pd.read_csv('C:/Users/tr5568/Desktop/DAYANA/CAPSA/csv/Dayana.csv', sep=",", parse_dates=[2])
#print(data)

for file_ in allFiles:
    data=pd.read_csv(file_,index_col=None, header=0,sep=",", parse_dates=[2])
    #BASELINE
    KL = np.array(data.loc[:,"KL_DETREND"])
    KL = savitzky_golay(KL, 61, 1)  # window size 51, polynomial order 3
    #plt.plot(KL)
    #plt.plot(data.loc[:, "KL_DETREND"])
    #plt.ylabel('some numbers')
    #plt.show()
    data["BASELINE"]=KL


    #Incremental KL_DETREND
    data["VENTA INCREMENTAL"]=data.loc[:,"KL_DETREND"]-data.loc[:,"BASELINE"]
    #print(data)

    #We read promotion file
    promo_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\PROMOCIONES_EROSKI_LYB_DDLL_2015_1710_II.xlsx"
    promo = pd.read_excel(promo_file)

    #We need 5 arrays in order to keep 3x"ANIMACIÓN", "ABREVIATURA ACCIÓN", "TEMÁTICA"
    #print(promo)
    animacion1=[]
    animacion2=[]
    animacion3=[]
    abr_accion=[]
    tematica=[]
    ventapromo=[]
    eurospromo=[]

    #We read canib file
    canib_file = "C:\\Users\\tr5568\\Desktop\\Dayana\\CAPSA\\Canib.xlsx"
    canib_excel = pd.read_excel(canib_file)
    #print(canib_excel)

    data=data.join(canib_excel.set_index('Cod. Familia'), on='FAMAPO')
    data.replace({'Cod. Familia':{"NaN":-1}}, inplace=True)
    print(data)



    for i in range(0,len(data.index)):
        #print(i)
        row_data=data.loc[i,:]
        #print(row_data)
        value=False
        #print(type(row_data[3]))
        #print(row)
        #print(len(row_data[3]))
        for j in range(0, len(promo.index)):
            row_promo=promo.loc[j,:]

            #if(row_data[3]==row_promo[2]): print("HOLA DAYI")
            #print(len(row_promo[2]))
            #print(row_promo)
            #print(str(row_promo[7]))
            if (not np.math.isnan(row_promo[7])):
                if(row_data[3]==row_promo[2] and row_data[4]==int(row_promo[7])):
                    #print("DAYI")
                    #print("ENSEÑA Y FAMILIA APO")
                    #print(row_data[3])
                    #print(row_promo[2])
                    #print(row_data[4])
                    #print(row_promo[7])
                    if(row_data[2]<=row_promo[4] and row_data[2]>=row_promo[3]):
                        #print("FECHAS")
                        #print(row_data[2])
                        #print(row_promo[4])
                        #print(row_promo[3])
                        value=True
                        row_aux=j

        if(value):
            #print("FIESTA")
            #print(promo.iloc[row_aux]["Animacion 1"])
            animacion1.append(promo.iloc[row_aux]["Animacion 1"])
            animacion2.append(promo.iloc[row_aux]["Animacion 2"])
            animacion3.append(promo.iloc[row_aux]["Animacion 3"])
            abr_accion.append(promo.iloc[row_aux]["Abreviatura accion"])
            tematica.append(promo.iloc[row_aux]["TEMATICA"])
            ventapromo.append(data.iloc[i]["KL_DETREND"])
            eurospromo.append(data.iloc[i]["EUROS_DETREND"])

        else:
            animacion1.append(0)
            animacion2.append(0)
            animacion3.append(0)
            abr_accion.append(0)
            tematica.append(0)
            ventapromo.append(0)
            eurospromo.append(0)


    #print(len(animacion1))
    data["ANIMACION1"]=animacion1
    data["ANIMACION2"]=animacion2
    data["ANIMACION3"]=animacion3
    data["ABREVIATURA_ACCION"]=abr_accion
    data["TEMÁTICA"]=tematica
    data["VENTA_PROMO"]=ventapromo
    data["EUROS_PROMO"]=eurospromo

    #print(data)

    with open("data.csv","a") as f:
        data.to_csv(f, sep=",",header=False)






