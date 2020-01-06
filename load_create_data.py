from skbox.connectors.bigquery import BigQuery
import pandas as pd 
import numpy as np


#selection de stypes 
sql = """
select NUM_ART,NUM_ETT,flag1 from `big-data-dev-lmfr.supply_EU.df_reappro1`
where  1 = 1 and
COD_RAY = 13 and 
COD_SRAY =5  and 
COD_TYP = 10 and 
COD_STYP = 10
and COD_RAY is not null
"""
bq = BigQuery()
df_reappro1 = bq.select(sql)
#TODO  : use a function to get combination between couple of MAG  (order is not important) (matrice triangulaire / superieur)
# import itertools
# comb = itertools.combinations(df_reappro1.NUM_ETT, 2)

#TODO : a simple test to optimize calculation ( dont do next steps if there's no commun product_typereappro 1 between 2 mag )

#all combinations
list_mag = df_reappro1.NUM_ETT.unique()
from itertools import combinations 
def rSubset(arr, r): 
  
    # return list of all subsets of length r 
    # to deal with duplicate subsets use  
    # set(list(combinations(arr, r))) 
    return list(combinations(arr, r))
couple_mag = rSubset(list_mag, 2)
len(couple_mag) #pour confirmer le rÃ©sultat de la fonction  Cn,p



def calcul_commun_products_reappro1(i) :
#TODO delete les NUM_ART (service / ceux qui commencent par 48/49)
    df_test = df_reappro1[df_reappro1.NUM_ETT.isin(i) ]
    df_test_flag1 = df_test[df_test.flag1 == 1]
    commun_products_flag1 = sum(sum(([df_test_flag1[["NUM_ART","flag1"]].groupby(["NUM_ART"]).size()==2])))
    return(commun_products_flag1)

def calcul_all_products(i) :
    df_test = df_reappro1[df_reappro1.NUM_ETT.isin(i) ]
    nb_all = len(df_test.NUM_ART.unique())
    return(nb_all)

df_result = pd.DataFrame()
df_result["couple_mag"] = couple_mag
df_result["union_product"] = df_result["couple_mag"].apply(lambda x: calcul_all_products(x))
df_result["commun_reappro1"] = df_result["couple_mag"].apply(lambda x: calcul_commun_products_reappro1(x))
df_result.to_csv("stype.csv",sep=";" , index = False)

#TODO :faire le mapping pour accÃ©eler le calcul

ex = df_result.copy()
l = ex['couple_mag'].astype("str").apply(lambda x: pd.Series(x.split(',')))
df_result["mag1"] = l[0].astype("str").apply(lambda x: int(re.search(r'\d+', x).group(0)))
df_result["mag2"]= l[1].astype("str").apply(lambda x: int(re.search(r'\d+', x).group(0)))

df_result["mag1"] = min(df_result["mag1"],df_result["mag2"])
df_result["mag2"] = max(df_result["mag1"],df_result["mag2"])
df_result["mag1"] = df_result[["mag1", "mag2"]].max(axis=1)
df_result["mag2"] = df_result[["mag1", "mag2"]].min(axis=1)
