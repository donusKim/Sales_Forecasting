import sys
import os
import numpy as np
import pandas as pd
import gc
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
from datetime import date, timedelta

## CSV파일 불러오기
df=pd.read_csv("./gdrive/My Drive/롯데백화점/purchase_data.csv",encoding="ms949" )
temp=pd.read_csv("./gdrive/My Drive/롯데백화점/기온2.csv",encoding="ms949").set_index(["지점"])
rain=pd.read_csv("./gdrive/My Drive/롯데백화점/강수량2.csv",encoding="ms949").set_index(["지점"])
point=pd.read_csv("./gdrive/My Drive/롯데백화점/point_table.csv",encoding="ms949").set_index(["point_name"])
trend=pd.read_csv("./gdrive/My Drive/롯데백화점/naver_trend.csv")

## Drop
temp.drop("160",axis=1,inplace=True)
rain.drop("160",axis=1,inplace=True)

###### 160주간의 데이터를 활용할것이고,,,시작날짜는 2016년 7월 3일, 그리고 지점NO별 지점명을 설정해준다
period=160
startDate='2016-07-03'
substitutions = {1.0: "본점",2.0: "잠실점",5.0: "부산본점",6.0: "관악점",8.0: "분당점",10.0: "영등포점",11.0: "일산점",13.0: "강남점",17.0: "창원점",22.0: "노원점",28.0: "건대스타점",333.0: "광복점"
                 ,341.0: "평촌점",344.0: "인천터미널점"}

## 전처리 함수 불러오기
from preprocess import mainPreprocess
from preprocess import weatherPreprocess
from preprocess import trendPreprocess

newDf=mainPreprocess(df,startDate,substitutions,period)
newRain=weatherPreprocess(newDf,rain,point)
newTemp=weatherPreprocess(newDf,temp,point)
newTrend=trendPreprocess(newDf,trend)

####### 3년내내 있는 브랜드는 몇개없다,,,,,160주동안 매출이 없는주 가 10주 이하인 브랜드 총 44개만 사용한다! (10개이하일경우 3년내내 입점한 브랜드라 가정해도 무방)
newTrend=newTrend.loc[newDf[(newDf==0).sum(axis=1).values<11].index]
newTemp=newTemp.loc[newDf[(newDf==0).sum(axis=1).values<11].index]
newRain=newRain.loc[newDf[(newDf==0).sum(axis=1).values<11].index]
newDf=newDf[(newDf==0).sum(axis=1).values<11]

## dataset 생성 함수 불러오기
from util import create_dataset
from util import train_generator

timesteps = 65
train_data = train_generator(newDf,newTemp,newRain,newTrend, timesteps, date(2019, 6, 2),n_range=13, batch_size=40)
Xval, Yval = create_dataset(newDf, newTemp,newRain,newTrend, timesteps, date(2019, 6,30))
Xtest, _= create_dataset(newDf, newTemp,newRain,newTrend, timesteps, date(2019, 7,28))

###### 모델 파라미터 설정
filterN=16
layerN=16
dr=0.3

## 모델 불러오기
from conv1d import makeModel
model=makeModel(filterN,layerN,dr,timesteps)

lr=0.0002
steps_per_epoch=40
workers=4
epochs=5
verbose=1

optimizer = optimizers.Adam(lr=lr)
model.compile(optimizer=optimizer, loss='mean_squared_error')

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

####훈련
history = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, workers=workers, use_multiprocessing=True, epochs=epochs, verbose=verbose,
                    validation_data=(Xval, Yval))

## 모델요약
model.summary()

## 텐서보드 시각화
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

## 예측실행
test_pred = model.predict(Xtest)

## 모델 저장 및 불러오기
model.save("sampleModel")
reconstructed_model = keras.models.load_model("sampleModel")

