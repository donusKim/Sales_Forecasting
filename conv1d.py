
import tensorflow as tf
import datetime
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers

def makeModel(filter_num,layer_num,dropout_rate,timesteps):
  ### timestep이 12라하면 예측이 시작되는 주 직전 12주간의 매출기록, 온도기록, 강수량기온, 검색량, 1년전 매출량,1분기전 매출량, 직전 12주간 브랜드 평균 매출,, 지점 평균매출,,,
  seq_in = Input(shape=(timesteps, 1))
  temp_in = Input(shape=(timesteps, 1))
  rain_in = Input(shape=(timesteps, 1))
  trend_in = Input(shape=(timesteps, 1))
  year_ago_in = Input(shape=(timesteps+4, 1))
  quarter_ago_in = Input(shape=(timesteps+4, 1))
  brand_mean_in = Input(shape=(timesteps, 1))
  point_mean_in = Input(shape=(timesteps, 1))

  encode_slice = Lambda(lambda x: x[:, :timesteps, :])
  encode_features = concatenate([ year_ago_in, quarter_ago_in], axis=2)
  encode_features = encode_slice(encode_features)
  x_in = concatenate([seq_in, encode_features, brand_mean_in,point_mean_in,trend_in,rain_in,temp_in], axis=2)
  

  c1 = Conv1D(filter_num, 3, dilation_rate=1, padding='causal', activation='relu')(x_in)
  conv_out = Dropout(dropout_rate)(c1)
  conv_out = Flatten()(conv_out)
  dnn_out = Dense(layer_num, activation='relu')(Flatten()(seq_in))
  dnn_out = Dropout(dropout_rate)(dnn_out)
  x = concatenate([conv_out, dnn_out])
  x = Dense(layer_num, activation='relu')(x)
  x = Dropout(dropout_rate)(x)
  output = Dense(4, activation='relu')(x)
  model = Model([seq_in, temp_in,rain_in,trend_in, year_ago_in, quarter_ago_in, brand_mean_in, point_mean_in], output)
  return model

