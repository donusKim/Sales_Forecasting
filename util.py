import numpy as np
import pandas as pd
from datetime import date, timedelta
import gc 

def create_dataset(df,temp,rain,trend, timesteps, pred_start, is_train=True):

    brand_mean_df = df.groupby('brand').mean().reindex(df.index.get_level_values(1))
    point_mean_df = df.groupby('point').mean().reindex(df.index.get_level_values(0))
   
    X, y = create_xy_span(df, pred_start, timesteps, is_train)
  
    rain = rain[pd.date_range(pred_start-timedelta(weeks=timesteps), periods=timesteps,freq='W')].values
    temp = temp[pd.date_range(pred_start-timedelta(weeks=timesteps), periods=timesteps,freq='W')].values
    trend = trend[pd.date_range(pred_start-timedelta(weeks=timesteps), periods=timesteps,freq='W')].values
    
    brand_mean, _ = create_xy_span(brand_mean_df, pred_start, timesteps, False)
    point_mean, _ = create_xy_span(point_mean_df, pred_start, timesteps, False)
    
    yearAgo, _ = create_xy_span(df, pred_start-timedelta(weeks=52), timesteps+4, False)
    quarterAgo, _ = create_xy_span(df, pred_start-timedelta(weeks=13), timesteps+4, False)

    rain = rain.reshape(-1, timesteps, 1)
    temp = temp.reshape(-1, timesteps, 1)
    trend = trend.reshape(-1, timesteps, 1)    
    brand_mean = brand_mean.reshape(-1, timesteps, 1)
    point_mean = point_mean.reshape(-1, timesteps, 1)
    yearAgo = yearAgo.reshape(-1, timesteps+4, 1)
    quarterAgo = quarterAgo.reshape(-1, timesteps+4, 1)

    return ([X,temp,rain, trend, yearAgo, quarterAgo,brand_mean, point_mean], y)


def train_generator(df,temp,rain,trend, timesteps, first_pred_start,
    n_range=1, is_train=True, batch_size=2000):

    while 1:
        date_part = np.random.permutation(range(n_range))
        ## df.shape[0]은 전체 brand개수,, batch_size는 44개중에서 몇개를 random하게 뽑아서 train dataset을 구성할지,,,
        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx,:]
            temp_tmp = temp.iloc[keep_idx,:]
            rain_tmp = rain.iloc[keep_idx,:]
            trend_tmp = trend.iloc[keep_idx,:]
            
            ## 첫 예측시작 시점으로부터 n_range 기간 앞부터 예측시작,,,뭔말이냐면 2019년 6월이 예측시작시점이고 n_range가 100주라면 100주전 99주전 98주전,,,0주전을 예측시점으로 잡고 set구성,,모든건 주단위
            pred_start = first_pred_start - timedelta(weeks=int(i))

            yield create_dataset(df_tmp,temp_tmp, rain_tmp, trend_tmp, timesteps, pred_start, is_train)
            gc.collect()


def create_xy_span(df, pred_start, timesteps, is_train=True, shift_range=0):
    X = df[pd.date_range(pred_start-timedelta(weeks=timesteps), pred_start-timedelta(weeks=1),freq='W')].values
    ## 훈련데이터라면,,,y생성하구 그게아니라면 y=None
    if (is_train & (date(2019,7,7)>pred_start)): y = df[pd.date_range(pred_start, periods=4,freq='W')].values
    else: y = None
    return X, y

