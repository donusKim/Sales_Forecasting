import sys
import os
import argparse
import numpy as np
import pandas as pd
from keras.layers import *
from keras import optimizers
from datetime import date
from preprocess import main_preprocess
from preprocess import weather_preprocess
from preprocess import trend_preprocess
from util import create_dataset
from util import train_generator
from conv1d import makeModel

if __name__ == '__main__':
    # 인자 받기
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", help="number of timesteps when making training dataset", default=52, type=int)
    parser.add_argument("--filter_num", help="number of filter in CNN Architecture", default=16, type=int)
    parser.add_argument("--layer_num", help="number of layer in CNN Architecture", default=16, type=int)
    parser.add_argument("--dropout_rate", help="dropout rate", default=0.3, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0005, type=int)
    parser.add_argument("--steps_per_epoch", help="steps per epoch", default=40, type=int)
    parser.add_argument("--epochs", help="number of epochs", default=5, type=int)
    args = parser.parse_args()

    # 파라미터 설정
    timesteps = args.timesteps
    filter_num = args.filter_num
    layer_num = args.layer_num
    dropout_rate = args.dropout_rate
    lr = args.lr
    steps_per_epoch = args.steps_per_epoch
    epochs = args.epochs

    ## Csv파일 불러오기
    df = pd.read_csv("purchase_data.csv", encoding="ms949")
    temp = pd.read_csv("temp.csv", encoding="ms949").set_index(["지점"])
    rain = pd.read_csv("rain.csv", encoding="ms949").set_index(["지점"])
    point = pd.read_csv("point_table.csv", encoding="ms949").set_index(["point_name"])
    trend = pd.read_csv("naver_trend.csv")

    ## Drop
    temp.drop("160", axis=1, inplace=True)
    rain.drop("160", axis=1, inplace=True)
    period = 160
    start_date = '2016-07-03'
    substitutions = {1.0: "본점", 2.0: "잠실점", 5.0: "부산본점", 6.0: "관악점", 8.0: "분당점", 10.0: "영등포점", 11.0: "일산점", 13.0: "강남점",
                     17.0: "창원점", 22.0: "노원점", 28.0: "건대스타점", 333.0: "광복점"
        , 341.0: "평촌점", 344.0: "인천터미널점"}

    ## 전처리
    new_df = main_preprocess(df, start_date, substitutions, period)
    new_rain = weather_preprocess(new_df, rain, point)
    new_temp = weather_preprocess(new_df, temp, point)
    new_trend = trend_preprocess(new_df, trend)

    ####### 3년내내 있는 브랜드는 몇개없다,,,,,160주동안 매출이 없는주 가 10주 이하인 브랜드 총 44개만 사용한다! (10개이하일경우 3년내내 입점한 브랜드라 가정해도 무방)
    new_trend = new_trend.loc[new_df[(new_df == 0).sum(axis=1).values < 11].index]
    new_temp = new_temp.loc[new_df[(new_df == 0).sum(axis=1).values < 11].index]
    new_rain = new_rain.loc[new_df[(new_df == 0).sum(axis=1).values < 11].index]
    new_df = new_df[(new_df == 0).sum(axis=1).values < 11]

    ## dataset 생성 함수 불러오기
    train_data = train_generator(new_df, new_temp, new_rain, new_trend, timesteps, date(2019, 6, 2), n_range=40,
                                 batch_size=30)
    x_val, y_val = create_dataset(new_df, new_temp, new_rain, new_trend, timesteps, date(2019, 6, 30))
    x_test, _ = create_dataset(new_df, new_temp, new_rain, new_trend, timesteps, date(2019, 7, 28))

    ## 모델 불러오기
    model = makeModel(filter_num, layer_num, dropout_rate, timesteps)
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    ####훈련
    history = model.fit(train_data, steps_per_epoch=steps_per_epoch, workers=1, use_multiprocessing=False,
                        epochs=epochs, verbose=1,
                        validation_data=(x_val, y_val))

    ## 모델요약
    # model.summary()

    ## 예측실행
    test_pred = model.predict(x_test)
    np.savetxt(os.getcwd() + '/preds/pred.csv', test_pred, delimiter=',')
    ## 모델 저장 및 불러오기
    model.save("sampleModel")
