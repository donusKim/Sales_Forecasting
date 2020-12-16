# Sales_Forecasting
## 1. 필요한 패키지
- [Keras](https://keras.io/ko/) - 2.3.1 이상의 version
- [Tensorflow](https://www.tensorflow.org/?hl=ko) - 2.0.0 이상의 version
## 2. 데이터
- 구매 데이터: purchase_data.csv
- 기온 데이터: 기온.csv
- 강수량 데이터: 강수량.csv
- 지점 데이터: point_table.csv
- 네이버 검색량 데이터: naver_trend.csv
- 구조 : dir tree 작성해야함
## 3. 코드
- [conv1d.py](https://github.com/donusKim/Sales_Forecasting/blob/master/conv1d.py)
   - conv1d모델에 관한 코드
- [preprocess.py](https://github.com/donusKim/Sales_Forecasting/blob/master/preprocess.py)
   - 전처리함수에 대한 코드
- [util.py](https://github.com/donusKim/Sales_Forecasting/blob/master/util.py)
   - 훈련데이터셋을 만드는 코드
- [main.py](https://github.com/donusKim/Sales_Forecasting/blob/master/main.py)
   - 데이터 불러오기,전처리,학습,추론이 이뤄지는 main 코드
## 4. 코드실행방법
- python main.py args들,,설명넣기,,
