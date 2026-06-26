# -*- coding: UTF-8 -*-

import json
import pandas as pd
import xgboost as xgb
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
# from dask import array as pd
# from dask.distributed import Client

# # It's recommended to use dask_cuda for GPU assignment
# from dask_cuda import LocalCUDACluster

# import cudf
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from xgboost import dask as dxgb
except ImportError:
    dxgb = None
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import sys 
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from biomass.geothermal.plotting import plotPredictedTest, plot_corr_matrix, plot_feature

class xgboostPro:
    def __init__(self,args):
        self.args = args
        self.params_algorithm = self.args.params_algorithm
        self.init_params()
        self.init_model()

    def init_model(self):        
        xgbReg = xgb.XGBRegressor(objective='reg:squarederror', seed=27, device=self.args.device, tree_method=self.args.tree_method)
        # xgbReg = xgb.XGBRegressor(objective='reg:squarederror', seed=27)    
        if 'grid' == self.params_algorithm:
            self.GSxgb = GridSearchCV(xgbReg, self.model_params, cv=5, verbose=10)
        elif 'random' == self.params_algorithm:
            self.GSxgb = RandomizedSearchCV(xgbReg,self.model_params, cv=5, verbose=10)
        else:
            self.GSxgb = xgbReg
    
    def init_params(self):
        self.model_params = {
        'max_depth': [12, 15, 18],
        'n_estimators': [1500],
        #'learning_rate': [0.01, 0.05],
        "gamma":[0.3, 0.5],
        # "reg_alpha":[0, 0.1, 0.5, 1],
        "reg_lambda":[0.3, 0.5, 0.7],
        "min_child_weight": [2, 3, 5],
        "colsample_bytree": [0.3, 0.6, 0.9],
        "subsample":[0.8, 0.9],
        "eta": [0.01, 0.05, 0.1]
        }    

    @staticmethod
    def load_data(data_path,Attempt,Run,is_land=False,use_bpnet=False, use_hypergbm=False):
        #read data
        df_o = pd.read_csv(data_path)
        df_o = df_o.dropna(subset=(['gradient'])) 
        # df_o = df_o.dropna(subset=(['binned-SL2013-2deg'])) 
        # df_o = df_o.fillna(df_o.mean()) 

        #land or ocean
        df_o = df_o[(df_o["is_land"]==is_land)]
        print(df_o.head(10))
        df_o = df_o.drop(["is_land"], axis=1)

        #data_preprocessing
        #check for nulls
        #cor_analysis & remove features which have higher than 0.8 cor_value with some others
        corr_matrix = df_o.corr(method='spearman')
        print(corr_matrix)
        plot_corr_matrix(corr_matrix,Attempt,Run)
        
        target = 'gradient'
        cols_pair_to_drop = []
        for index_ in corr_matrix.index:
            for col_ in corr_matrix.columns:
                if target in (index_, col_):
                    continue
                if corr_matrix.loc[index_,col_] >= 0.8 and index_!=col_ and (col_,index_) not in cols_pair_to_drop:
                    cols_pair_to_drop.append((index_,col_))
        # print("cols_pair_to_drop:", cols_pair_to_drop)
        cols_to_drop = np.unique([col[1] for col in cols_pair_to_drop])  
        df_o.drop(cols_to_drop,axis=1,inplace=True)

        #remove repitition & shuffle
        # df = df_o.drop_duplicates()
        df = shuffle(df_o)
        #remove abnormal values
        items = list(df.columns.values)
        for i in items:
            plt.figure()
            box_plt = df.boxplot(column=[i])
            box_plt = box_plt.get_figure()
            box_plt.savefig("%sAttempt/%s/box_plt_%s.png" % (Attempt,Run,i))
        
        #filter
        try:
            iqr = df[target].quantile(0.75) - df[target].quantile(0.25)
            q_abnormal_L = df[target] < df[target].quantile(0.25) - 1.5 * iqr
            q_abnormal_U = df[target] > df[target].quantile(0.75) + 1.5 * iqr

            print(target + ' has' + str(q_abnormal_L.sum() + q_abnormal_U.sum()) + ' abnormal values')
            item_outlier_index = df[q_abnormal_L|q_abnormal_U].index

            df.drop(index = item_outlier_index,inplace=True)
            df = df.dropna(subset=(['gradient']))
            # df[target] = df[target].fillna(df[target].median())

        except:
            print("Error in filter!!!")

        if Attempt is None or Run is None:
            raise ValueError("Attempt and Run must be provided before loading data.")

        cli_args = globals().get("args")
        if cli_args is not None and cli_args.standardscaler:
            scaler = MinMaxScaler()
            df[items] = scaler.fit_transform(df[items])
            # print(df.shape)

        items.remove("gradient")
        # items.remove("is_land")
        features = items
        df.to_csv(f"{Attempt}Attempt/%s/final_data.csv"%Run, index=False)
        X=pd.DataFrame(df, columns=features)
        y=pd.DataFrame(df["gradient"])        
        # pd_X=pd.DataFrame(df, columns=features)
        # pd_y=pd.DataFrame(df["gradient"])
        # np_X = pd_X.to_numpy()
        # np_y = pd_y.to_numpy()
        # X = cp.asarray(np_X)
        # y = cp.asarray(np_y)

        if  use_bpnet:
            X = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
            y = torch.from_numpy(np.array(y)).type(torch.FloatTensor)

        # elif  use_hypergbm:
        # else:
        #     X=cp.asarray(X.to_numpy())
        #     y=cp.asarray(y.to_numpy())
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
        x_train.to_csv("%sAttempt/%s/x_train_data.csv" % (Attempt,Run), index=False)
        y_train.to_csv("%sAttempt/%s/y_train_data.csv" % (Attempt,Run), index=False)
        x_test.to_csv("%sAttempt/%s/x_test_data.csv" % (Attempt,Run), index=False)
        y_test.to_csv("%sAttempt/%s/y_test_data.csv" % (Attempt,Run), index=False)
        return x_train, x_test, y_train, y_test, features           
    
    def train(self,x_train,y_train,Attempt,Run,params_algorithm):
        print("xgboost started training")
        self.GSxgb.fit(x_train, y_train)
        cv_results = pd.DataFrame(self.GSxgb.cv_results_)
        cv_results.to_csv("%sAttempt/%s/xgboost_%s_cv_results.csv" % (Attempt,Run,params_algorithm), index=False)
        return self.GSxgb
    
    @staticmethod 
    def test(model,params_algorithm,x_train,y_train,x_test,y_test,Attempt,Run,features):
        model_score=model.score(x_train,y_train)
        y_pred=model.predict(x_test)
        RMSE=root_mean_squared_error(y_test,y_pred)
        # r2=r2_score(y_test.to_numpy(),y_pred.to_numpy())
        r2=r2_score(y_test,y_pred)        

        # save_result
        if params_algorithm in ['grid','random']:
            Best = model.best_params_
            NE = Best['n_estimators']
            # LR = Best['learning_rate']
            MD = Best['max_depth']
            # SS = Best['min_samples_split']
            Su = Best['subsample']
            Gamma = Best['gamma']
            # RA = Best['reg_alpha']
            RL = Best['reg_lambda']
            MW = Best['min_child_weight']
            CB = Best['colsample_bytree']
            ETA = Best['eta']
        else:
            LR = 0.01
            MD = 11
            SS = 5
            Su = 0.7     
        
        M=np.vstack((model_score, r2, RMSE, MD, Su, NE, ETA, MW, Gamma, CB, RL)).T
        np.savetxt('%sAttempt/%s/Scores_%s.txt' % (Attempt,Run,Run), M, fmt='%.3f', 
                header='modelScore r2 RMSE MaxDepth subsample n_estimators eta min_child_weight gamma colsample_bytree reg_lambda')

        y_pred = y_pred[:,np.newaxis]
        Corr=np.hstack((y_test, y_pred))
        np.savetxt('%sAttempt/%s/Y_%s.txt' % (Attempt,Run,Run), Corr, fmt='%.3f', 
                header='Actual Predicted')

        print('model score:', model_score)
        print('Test Variance score (r2):', r2)
        print('RMSE:', RMSE)
        plotPredictedTest(y_test,y_pred,Attempt,Run)
        plot_feature(model.best_estimator_.feature_importances_,features,Attempt,Run)

    @staticmethod 
    def error_estimation(model,params_algorithm,x_train,y_train,x_test,y_test,Attempt,Run,features):
        '''to calculate the error for all the data
        '''
        model_score=model.score(x_train,y_train)
        X = pd.concat([x_train,x_test], axis=0)
        y = pd.concat([y_train,y_test], axis=0)
        y_pred=model.predict(X)
        RMSE=root_mean_squared_error(y,y_pred)
        r2=r2_score(y,y_pred)

        # save_result
        if params_algorithm in ['grid','random']:
            Best = model.best_params_
            NE = Best['n_estimators']
            # LR = Best['learning_rate']
            MD = Best['max_depth']
            # SS = Best['min_samples_split']
            Su = Best['subsample']
            Gamma = Best['gamma']
            # RA = Best['reg_alpha']
            RL = Best['reg_lambda']
            MW = Best['min_child_weight']
            CB = Best['colsample_bytree']
            ETA = Best['eta']
        else:
            LR = 0.01
            MD = 11
            SS = 5
            Su = 0.7     
        
        M=np.vstack((model_score, r2, RMSE, MD, Su, NE, ETA, MW, Gamma, CB, RL)).T
        np.savetxt('%sAttempt/%serror/Error_Scores_%s.txt' % (Attempt,Run,Run), M, fmt='%.3f', 
                header='modelScore r2 RMSE MaxDepth subsample n_estimators eta min_child_weight gamma colsample_bytree reg')

        y_pred = y_pred[:,np.newaxis]
        Corr=np.hstack((y, y_pred))
        np.savetxt('%sAttempt/%s/Error_Y_%s.txt' % (Attempt,Run,Run), Corr, fmt='%.3f', 
                header='Actual Predicted')

        print('model score:', model_score)
        print('Test Variance score (r2):', r2)
        print('RMSE:', RMSE)
        plotPredictedTest(y,y_pred,Attempt,Run+'error') #Run+0.1 used to be discriminated from the test pic
        plot_feature(model.best_estimator_.feature_importances_,features,Attempt,Run)

    def save_model(self):
        model_s = pickle.dumps(self.GSxgb)
        # save model
        with open('%sAttempt/%s/myModel%s.model'%(self.args.Attempt,self.args.Run,self.args.Attempt),'wb+') as f:#'wb+'means written in binary
            f.write(model_s)
        f.close() 
    
    @staticmethod  # test local model 
    def load_model(local_model):
        f = open(local_model,'rb') 
        s = f.read()
        model = pickle.loads(s)
        return model
    
    @staticmethod
    def data_inference(file_path):
        '''return oceanic data and continental data in order
        '''
        #also use args.data_path
        df = pd.read_csv(file_path, index_col=None)
        df_o = df[(df["is_land"]==False)]
        df_c = df[(df["is_land"]==True)]
        df_io = df_o[df_o['gradient'].isna()]
        df_nio = df_o[df_o['gradient'].notna()]
        df_ic = df_c[df_c['gradient'].isna()]
        df_nic = df_c[df_c['gradient'].notna()]
        return df_io, df_nio, df_ic, df_nic

    def poltting(self):
        ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Attempt', type=str, default='1st')
    parser.add_argument('--Run', type=str, default='continental')
    parser.add_argument('--Res', type=str, default= '01')
    parser.add_argument('--run_type', type=str, default= 'train')
    parser.add_argument('--params_algorithm', type=str, default='random',choices=['grid','random',None])
    parser.add_argument('--data_path', type=str, default= '../data/geothermal_model_final_data/split_ocean_1x1.csv')
    # parser.add_argument('--gridsearch', type=bool, default=True)
    parser.add_argument('--standardscaler', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tree_method', type=str, default='hist')    
    parser.add_argument('--BPnet', type=bool, default=False)
    parser.add_argument('--is_land', action="store_true", default=False, help="choose data of continents or oceans")
    parser.add_argument('--if_inference', action="store_true", default=False, help="choose if use the trained model to inference")
    parser.add_argument('--omodel_path', default= '1stAttempt/oceanic_final/myModel1st.model', help="choose model for predicting the oceanic data")
    parser.add_argument('--cmodel_path', default= '1stAttempt/continental_final/myModel1st.model', help="choose model for predicting the continental data")
    
    args = parser.parse_args()
    args_dict = vars(args)
    if not os.path.exists(args.Attempt + 'Attempt/' + args.Run +'/'):
        os.makedirs(args.Attempt + 'Attempt/' + '/' + args.Run +'/' )
    if not os.path.exists(args.Attempt + 'Attempt/' + '/'  + args.Run +'/' + "Plots/"):
        os.makedirs(args.Attempt + 'Attempt/' + '/'  + args.Run +'/' + "Plots/") 
    if not os.path.exists(args.Attempt + 'Attempt/' + '/'  + args.Run +'/' + "Plots/"):
        os.makedirs(args.Attempt + 'Attempt/' + '/'  + args.Run +'/' + "Plots/") 
    if not os.path.exists(args.Attempt + 'Attempt/' + '/'  + args.Run +'error/' + "Plots/"):
        os.makedirs(args.Attempt + 'Attempt/' + '/'  + args.Run +'error/' + "Plots/") 

    with open('%sAttempt/%s/arguments.json'%(args.Attempt,args.Run), 'w') as file:
        json.dump(args_dict, file, indent=4)

    if args.if_inference:
        df_io, df_nio, df_ic, df_nic = xgboostPro.data_inference(args.data_path)
        ocean_model = xgboostPro.load_model(args.omodel_path)
        continental_model = xgboostPro.load_model(args.cmodel_path)
        ocean_features = ['RTP-BZ-400-05', 'Volcanos', 'YoungRift', 'LithoRef18-1deg', 'Trench', 'TopoI1', 'MeanCurvature-TopoIso-corr', 
                             'Susceptibility-Sz-LitMod-Aus17-Afr', 'Ridge',	'binned-SL2013-2deg', 'lon', 'lat',	'Depth2Moho', 'VP1', 'VP3']
        # ocean_features = ['LAB', 'Curie', 'elevation', 'Moho', 'Magnetic', 'Ridge', 'Susceptibility', 'tectonicunit', 'Transform', 'Volcanos', 
                        #   'YoungRift', 'VP1', 'VP2', 'VP3', 'VS3', 'lat', 'lon']
                          
        X_o = pd.DataFrame(df_io, columns=ocean_features)
        y_o= ocean_model.predict(X_o)
        df_io['gradient'] = y_o
        df_io.to_csv("%sAttempt/inference_oceanic.csv"%args.Attempt, index=False)
        # continental_features = ['RTP','LAB','Moho','Magnetic','Ridge','Susceptibility','tectonicunit','Topo','Transform',
        #                   'Volcanos','YoungRift','lat','lon']
        continental_features = ['RTP-BZ-400-05', 'Volcanos', 'YoungRift', 'LithoRef18-1deg', 'Trench', 'TopoI1', 'MeanCurvature-TopoIso-corr', 
                             'Susceptibility-Sz-LitMod-Aus17-Afr', 'Ridge',	'binned-SL2013-2deg', 'lon', 'lat',	'Depth2Moho', 'VP1', 'VS3']
        X_c = pd.DataFrame(df_ic, columns=continental_features)
        y_c= continental_model.predict(X_c)
        df_ic['gradient'] = y_c
        df_ic.to_csv("%sAttempt/inference_continental.csv"%args.Attempt, index=False)
        final_ocean = pd.concat([df_nio,df_io], ignore_index=True)
        final_continental = pd.concat([df_nic,df_ic], ignore_index=True)
        final_ocean.to_csv("%sAttempt/total_oceanic.csv"%args.Attempt, index=False)
        final_continental.to_csv("%sAttempt/total_continental.csv"%args.Attempt, index=False)
    
    elif args.BPnet == False:
        
        if not os.path.exists(args.Attempt + 'Attempt/' + args.Run +'/'):
            os.makedirs(args.Attempt + 'Attempt/' + args.Run +'/' )
        if not os.path.exists(args.Attempt + 'Attempt/' + args.Run +'/' + "Plots/"):
            os.makedirs(args.Attempt + 'Attempt/' + args.Run +'/' + "Plots/")
        
        if args.run_type == 'train':
            xgboost_pro = xgboostPro(args)
            xgboost_pro.init_model() # 
            x_train, x_test, y_train, y_test,features= xgboostPro.load_data(args.data_path,args.Attempt,args.Run,args.is_land)
            
            trained_model = xgboost_pro.train(x_train,y_train,args.Attempt,args.Run,args.params_algorithm)
            xgboost_pro.test(trained_model,args.params_algorithm,x_train,y_train,x_test,y_test,args.Attempt,args.Run,features)
            xgboost_pro.error_estimation(trained_model,args.params_algorithm,x_train,y_train,x_test,y_test,args.Attempt,args.Run,features)            
            xgboost_pro.save_model()
            
        elif args.run_type == 'test':
            x_train, x_test, y_train, y_test, features = xgboostPro.load_data(args.data_path,args.Attempt,args.Run,is_land=args.is_land)
            local_model = xgboostPro.load_model(args.cmodel_path)
            xgboostPro.test(local_model,args.params_algorithm,x_train,y_train,x_test,y_test,args.Attempt,args.Run,features)

    else:
        from BPNet.BPnet import *
        if not os.path.exists(args.Attempt + 'Attempt/' + args.Run +'/'):
            os.makedirs(args.Attempt + 'Attempt/' + args.Run +'/' )
        if not os.path.exists(args.Attempt + 'Attempt/' + args.Run +'/' + "Plots/"):
            os.makedirs(args.Attempt + 'Attempt/' + args.Run +'/' + "Plots/")       
        parser.add_argument('--model_list', type=list, default=[14,500,1000,1000,500,500,1])
        parser.add_argument('--epochs', type=int,default= 1000)
        parser.add_argument('--Learning_rate', type=float,default= 0.01)

        args = parser.parse_args()
        
        x_train, x_test, y_train, y_test,features = xgboostPro.load_data(args.data_path,args.Attempt,args.Run,is_land=args.is_land,use_bpnet=True)

        bp_net = bpNetPro(args)
        bp_net.init_model()
        if torch.cuda.is_available():
            bp_net.model.cuda()
            bp_net.train(x_train.cuda(),y_train.cuda())
        else:
            bp_net.train(x_train,y_train)
        bp_net.model.cpu()
        bp_net.test(bp_net.model,x_test,y_test)
