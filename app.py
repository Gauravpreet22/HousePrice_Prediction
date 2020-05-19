# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:03:30 2020

@author: Owner
"""
# import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
Ridge_model = pickle.load(open('Ridge_model.pkl', 'rb'))
Lasso_model = pickle.load(open('Lasso_model.pkl', 'rb'))
Linear_reg_model = pickle.load(open('Linear_reg_model.pkl', 'rb'))

def response_to_numeric(resp_arr, resp_value):
    response_list = []
    for resp in resp_arr:
        
        if (resp_value == resp):
            response_list = np.append(response_list, "1")
        else:
            response_list = np.append(response_list, "0")
    return response_list.astype(int)

# housing_df = pd.read_csv('D:/Reverse Engineering/House Price/train.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    output = []
    output = np.append(output, request.form.get('OverallQual'))
    output = np.append(output, request.form.get('ExterQual'))
    output = np.append(output, request.form.get('BsmtQual'))
    output = np.append(output, request.form.get('HeatingQC'))
    output = np.append(output, request.form.get('KitchenQual'))
    output = np.append(output, request.form.get('Fireplaces'))
    output = np.append(output, request.form.get('FireplaceQu'))
    output = np.append(output, request.form.get('GarageFinish'))  
    
    output = np.append(output, request.form.get('GarageCars'))
    output = np.append(output, request.form.get('TotalBathRooms'))
    output = np.append(output, request.form.get('YearsOld'))
    tarea = (float)(request.form.get('TotalArea'))
    tarea_log = np.log1p(tarea)
    output = np.append(output, tarea_log)
    # output = np.append(output, request.form.get('TotalArea'))
    #Condition
    Condition_sel_resp = request.form.get('Condition')
    Condition_Num_resp = response_to_numeric(["Artery","Feedr","Norm","PosA","PosN","RRAe","RRAn","RRNe","RRNn"],Condition_sel_resp)
    output = np.append(output, Condition_Num_resp)
    
    #Exterior
    Exterior_sel_resp = request.form.get('Exterior')
    Exterior_Num_resp = response_to_numeric(['VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','WdShing','CemntBd','Plywood','AsbShng','Stucco','BrkComm','AsphShn','Stone','ImStucc','CBlock'],Exterior_sel_resp)
    output = np.append(output, Exterior_Num_resp)

    # MSSubClass
    MSSubClass_sel_resp = request.form.get('MSSubClass')
    MSSubClass_Num_resp = response_to_numeric(['MSSubClass_160MS','MSSubClass_190MS','MSSubClass_20MS','MSSubClass_30MS','MSSubClass_45MS','MSSubClass_50MS','MSSubClass_60MS','MSSubClass_70MS','MSSubClass_75MS', 'MSSubClass_80MS','MSSubClass_85MS','MSSubClass_90MS'], MSSubClass_sel_resp)
    output = np.append(output, MSSubClass_Num_resp)
    
    # MSZoning
    MSZoning_sel_resp = request.form.get('MSZoning')    
    MSZoning_Num_resp = response_to_numeric(['MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM'], MSZoning_sel_resp)
    output = np.append(output, MSZoning_Num_resp)
    
    # Street_Pave
    output = np.append(output, request.form.get('street'))
    # alley_Pave
    output = np.append(output, request.form.get('alley'))
    
    #LandContour
    LandContour_sel_resp = request.form.get('LandContour')    
    LandContour_Num_resp = response_to_numeric(['LandContour_HLS','LandContour_Low','LandContour_Lvl'], LandContour_sel_resp)
    output = np.append(output, LandContour_Num_resp)
    
    #LotConfig
    LotConfig_sel_resp = request.form.get('LotConfig')
    LotConfig_Num_resp = response_to_numeric(["LotConfig_CulDSac", "LotConfig_FR2", "LotConfig_Inside"],LotConfig_sel_resp)
    output = np.append(output, LotConfig_Num_resp)
    
    #Neighborhood
    Neighborhood_sel_resp = request.form.get('Neighborhood')
    Neighborhood_Num_resp = response_to_numeric(["Neighborhood_BrDale","Neighborhood_BrkSide","Neighborhood_ClearCr","Neighborhood_CollgCr","Neighborhood_Crawfor","Neighborhood_Edwards","Neighborhood_Gilbert","Neighborhood_IDOTRR","Neighborhood_MeadowV","Neighborhood_Mitchel","Neighborhood_NAmes","Neighborhood_NWAmes","Neighborhood_NoRidge","Neighborhood_NridgHt","Neighborhood_OldTown","Neighborhood_SWISU","Neighborhood_Sawyer","Neighborhood_SawyerW","Neighborhood_Somerst","Neighborhood_StoneBr","Neighborhood_Timber","Neighborhood_Veenker"],Neighborhood_sel_resp)
    output = np.append(output, Neighborhood_Num_resp)
    
    #BldgType
    BldgType_sel_resp = request.form.get('BldgType')
    BldgType_Num_resp = response_to_numeric(["BldgType_2fmCon","BldgType_Duplex","BldgType_Twnhs","BldgType_TwnhsE"],BldgType_sel_resp)
    output = np.append(output, BldgType_Num_resp)
    
    #HouseStyle
    HouseStyle_sel_resp = request.form.get('HouseStyle')
    HouseStyle_Num_resp = response_to_numeric(['HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl'],HouseStyle_sel_resp)
    output = np.append(output, HouseStyle_Num_resp)
    
    #RoofStyle
    RoofStyle_sel_resp = request.form.get('RoofMatl')
    RoofStyle_Num_resp = response_to_numeric(['RoofStyle_Gable', 'RoofStyle_Gambrel','RoofStyle_Hip'],RoofStyle_sel_resp)
    output = np.append(output, RoofStyle_Num_resp)

    #RoofMatl
    RoofMatl_sel_resp = request.form.get('RoofMatl')
    RoofMatl_Num_resp = response_to_numeric(['RoofMatl_CompShg', 'RoofMatl_Tar&Grv'],RoofMatl_sel_resp)
    output = np.append(output, RoofMatl_Num_resp)    
    
    #MasVnrType
    MasVnrType_sel_resp = request.form.get('MasVnrType')
    MasVnrType_Num_resp = response_to_numeric(['MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone'],MasVnrType_sel_resp)
    output = np.append(output, MasVnrType_Num_resp)

    #Foundation
    Foundation_sel_resp = request.form.get('Foundation')
    Foundation_Num_resp = response_to_numeric(['Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab'],Foundation_sel_resp)
    output = np.append(output, Foundation_Num_resp)

    #Heating
    Heating_sel_resp = request.form.get('Heating')
    Heating_Num_resp = response_to_numeric(['Heating_GasA', 'Heating_GasW'],Heating_sel_resp)
    output = np.append(output, Heating_Num_resp)
    
    #CentralAir
    CentralAir_sel_resp = request.form.get('CentralAir')
    output = np.append(output, CentralAir_sel_resp)
    
    #Electrical
    Electrical_sel_resp = request.form.get('Electrical')
    Electrical_Num_resp = response_to_numeric(['Electrical_FuseF', 'Electrical_SBrkr'],Electrical_sel_resp)
    output = np.append(output, Electrical_Num_resp)
    
    #Functional
    Functional_sel_resp = request.form.get('Functional')
    Functional_Num_resp = response_to_numeric(['Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ'],Functional_sel_resp)
    output = np.append(output, Functional_Num_resp)
    
    #GarageType
    GarageType_sel_resp = request.form.get('GarageType')
    GarageType_Num_resp = response_to_numeric(['GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_Detchd', 'GarageType_none'],GarageType_sel_resp)
    output = np.append(output, GarageType_Num_resp)
    
    #Fence
    Fence_sel_resp = request.form.get('Fence')
    Fence_Num_resp = response_to_numeric(['Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw'],Fence_sel_resp)
    output = np.append(output, Fence_Num_resp)
    
    #MiscFeature_Shed
    MiscFeature_Shed_sel_resp = request.form.get('MiscFeature_Shed')
    output = np.append(output, MiscFeature_Shed_sel_resp)
    
    #SaleType
    SaleType_sel_resp = request.form.get('SaleType')
    SaleType_Num_resp = response_to_numeric(['SaleType_New', 'SaleType_WD'],SaleType_sel_resp)
    output = np.append(output, SaleType_Num_resp)
    
    #SaleCondition
    SaleCondition_sel_resp = request.form.get('SaleCondition')
    SaleCondition_Num_resp = response_to_numeric(['SaleCondition_Alloca', 'SaleCondition_Family','SaleCondition_Normal'],SaleCondition_sel_resp)
    output = np.append(output, SaleCondition_Num_resp)    
    
    prediction = Lasso_model.predict([output.astype(float)])
    # prediction = Linear_reg_model.predict([output.astype(float)])
    fin_pred = math.exp(prediction[0])
    # final_pred = round(prediction[0],2)
    # return render_template('index.html', prediction_text = 'Predicted price Of the house is USD{}'.format(tarea))
    return render_template('index.html', prediction_text = 'Predicted price Of the house is USD{}'.format(round(fin_pred,2)))
    
# return render_template('index.html', prediction_text = 'selected rating is {}'.format(RoofMatl_Num_resp))
    # init_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(init_features)]
    # prediction = model.predict(final_features)
    # output = round(prediction[0],2)
    # return render_template('index.html', prediction_text = 'predicted Salary is {}'.format(output))

if (__name__ == "__main__"):
    app.run(debug=True)
