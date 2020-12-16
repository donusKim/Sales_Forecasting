import numpy as np
import pandas as pd
import datetime

### 영수증 데이터 기록을 전처리하는 함수,,, 날짜별로 브랜드별로 판매금액을 표시하게됨,,
def main_preprocess(df,start_date,substitutions,period,week=True):
    ## datetime형식으로 변환,,주단위로 묶어서 하는이유,,,하루하루 빈매출이 너무많아서,,sparse해지는걸 방지
    df["date"]=df['date'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d'))
    df["date"]=df["date"]-np.datetime64(start_date)
    if(week):
        df["date"]=df["date"]/7 
    ## week에서 소수점 버리기!
    df["date"]=df["date"].apply(lambda x: x.days)
    ## 취소매출 제거하고,,필요한 데이터(지점,몇번째 주인지, 브랜드, 판매금액 뽑아온다)
    df["pur_amt"]=abs(df["pur_amt"])
    df=df.drop_duplicates(["cus_id","pur_amt"], keep='first')
    new_df=df[["point","date","brand","pur_amt"]]
    new_df["point"]=new_df["point"].replace(substitutions)
    ## 정해진 기간만큼의 데이터만 사용하기 때문에 하는 전처리
    new_df=new_df[(new_df["date"]<period) & (new_df["date"]>-1) ]
    #전처리 위해 reindex,,,점포별,,,주차별 브랜드별 판매금액을 나타내기위해!
    new_df=new_df.set_index(["point","brand","date"]).groupby(level=[0,1,2])['pur_amt'].agg({('sum',np.sum)}).unstack(level=-1).fillna(0)
    new_df.columns=new_df.columns.get_level_values(1)
    new_df=new_df[sorted(new_df.columns)]
    if(week):
        new_df.columns=pd.date_range(start=start_date, periods=period, freq='W')
    else:
        new_df.columns=pd.date_range(start=start_date, periods=period, freq='D')
        
    return new_df


### 강수량과 기온에 대해 전처리하는 함수,,,
def weather_preprocess(new_df,weather,point):
    ## 160주간 기온과 강수량,,,,,!!!!!!!!!!!!!! min max 스케일링을 한이유!! 찾아오기
    weather=weather.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    #인천제거,,,인천지점 분석에 사용하지 않기 떄문에!
    weather.drop("인천",axis=0,inplace=True)
    new_weather=pd.merge(point, weather,left_on='지역', right_index=True,how='left')
    new_weather.drop("지역",axis=1,inplace=True)
    new_weather=new_weather.reindex(new_df.index.get_level_values(0))
    new_weather.columns=new_df.columns
    new_weather.index=new_df.index
    return new_weather



### 네이버 검색량(Trend)에 대해 전처리하는 함수,,,
def trend_preprocess(new_df,trend):
    trend=trend.transpose().drop('날짜')
    trend=trend.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    trend=trend.reindex(new_df.index.get_level_values(1))
    trend.columns=new_df.columns
    trend.index=new_df.index
    return trend
