import pandas as pd

def Canibalizacion():
    data=pd.read_csv('C:/Users/tr5568/PycharmProjects/CAPSA/data.csv', encoding='ISO-8859-1')
    print(data)

    #unique=distinct
    #canib_vector=pd.unique(data['Grupo canibalizacion'])

    #group by Grupo canibalizacion and DATE
    data_canib=data.groupby(['Grupo canibalizacion', 'DATE'])

    #we have a new df with sum and size of BASELINE and KL_DETREND
    #we need this df to calculate canibalizacion
    #canibalizacion =
    #mean of KL_DETREND of all the products with the same Grupo canibalizacion (without the one we are considering) -
    # -mean of BASELINE of all the products with the same Grupo canibaliacion (without the one we are considering)
    df_baseline=data_canib['BASELINE'].agg(['sum','size']).reset_index()
    df_KLdetrend=data_canib['KL_DETREND'].agg(['sum','size']).reset_index()
    #print(df_baseline)

    canib=[]


    for i in range(0,len(data.index)):
        row_data = data.loc[i, :]
        #print(row_data)
        aux_base=df_baseline[(df_baseline['Grupo canibalizacion']==row_data[16]) & (df_baseline['DATE']==row_data[3])]
        aux_KLdetrend=df_KLdetrend[(df_KLdetrend['Grupo canibalizacion']==row_data[16]) & (df_KLdetrend['DATE']==row_data[3])]
        #print((aux_base['sum']-row_data[14])/(aux_base['size']-1)-((aux_KLdetrend['sum']-row_data[12])/(aux_KLdetrend['size']-1)))
        canib.append(float(-((aux_base['sum']-row_data[14])/(aux_base['size']-1))+((aux_KLdetrend['sum']-row_data[12])/(aux_KLdetrend['size']-1))))

    #add the new calculated column to our data
    data["CANIBALIZACION"]=canib

    #we write data in a csv file
    data.to_csv("data_canib.csv", sep=',')


