---
layout: post
title: "[데이터 분석가가 반드시 알아야할 모든 것] 데이터 탐색과 시각화"
published: true
date: 2024-07-23
math: true
categories: 
    - Study
    - Data Science
tags: KHUDA ML
---


> - EDA(Exploratory Data Analysis 탐색적 데이터 분석)와 데이터 시각화의 목적은 조금 다르다.
- EDA 단계에서 데이터 파악을 효율적으로 하기 위해 시각화를 하기도 하지만, 데이터 시각화의 궁극적인 목적은 분석 결과를 커뮤니케이션 하기 위함이다.

- **시각화의 유형**
1. 시간의 흐름에 따른 변화 -> 시간 시각화
2. 그룹별 차이 -> 비교 시각화
3. 전체 data에서 특정 항목의 비중 -> 분포 시각화
4. 2개 이상의 수치 data 간의 관계 -> 관계 시각화
5. 지리적 위치에 수치 표기 -> 공간 시각화

## **1. 탐색적 data 분석**
- EDA를 할 때에는 극단적인 해석, 지나친 추론, 자의적 해석을 지양해야 한다.

> **가장 간단하면서도 효과적인 방법 : 1000개의 data sample과 변수와 설명 list를 눈으로 보기**
- 논리적으로 잘못된 부분을 찾아내고, 해당 이슈가 발생할 수 있는 case를 고밍해본다.

> - EDA(Exploratory Data Analysis 탐색적 데이터 분석)와 데이터 시각화의 목적은 조금 다르다.
- DEA 단계에서 데이터 파악을 효율적으로 하기 위해 시각화를 하기도 하지만, 데이터 시각화의 궁극적인 목적은 분석 결과를 커뮤니케이션 하기 위함이다.

- **시각화의 유형**
1. 시간의 흐름에 따른 변화 -> 시간 시각화
2. 그룹별 차이 -> 비교 시각화
3. 전체 data에서 특정 항목의 비중 -> 분포 시각화
4. 2개 이상의 수치 data 간의 관계 -> 관계 시각화
5. 지리적 위치에 수치 표기 -> 공간 시각화


```python
df.info()
# 숫자형으로 되어야 하는 column이 문자형으로 되어있을 수도 있기에, 자료형을 확인
# 결측값 확인
```


```python
기술 통게
df.describe()
```

> **숫자형이지만, 문자형과 다름 없는 arrival_date_year와 같은 column들에 주의하자.**

## **2. 공분산과 상관성 분석**

- 타깃 변수 y와 입력 변수 x와의 관계 & 입력 변수들 간의 관계 확인
- 다중 공선성 방지 및 데이터에 대한 이해도 향상
- 등간(절대적인 원점이 존재 x) / 비율 척도(0값이 의미가 있음)인 data + 두 변수가 선형적 관계라는 가정을 두고 상관 분석을 진행

![111](https://github.com/user-attachments/assets/f055265b-2bb7-432d-9613-2dc2e85785b8)

- 공분산 : 두 변수가 서로 공유하는 분산
- 공분산 행렬을 계산할 때 사용되는 X : 원본 데이터를 그대로 사용하지 않고, 각 변수의 평균을 뺀 중심화된(centered) 데이터를 사용.
- 각 열은 해당 변수의 관측값에서 그 변수의 평균을 뺀 값들로 구성됩니다.
-  변수들 간의 선형 관계를 정확히 표현할 수 있으며, 동시에 각 변수의 분산도 대각선 원소로 얻을 수 있습니다.
- 공분산 계산식 : $Cov(X,Y)=E[(X−E[X])(Y−E[Y])]$
  
- 공분산 행렬 :

 ```math
\Sigma = \frac{1}{n-1} \mathbf{X}_\text{centered}^T \mathbf{X}_\text{centered}
 ```

 ```math
 \Sigma_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (X_{ki} - \bar{X}_i)(X_{kj} - \bar{X}_j)
 ```

>- 두 변수가 같은 방향으로 변할 때 (함께 증가하거나 함께 감소할 때): 공분산 값은 양수(+)가 됩니다. -> 두 변수 사이에 양의 관계가 있음을 의미
- 두 변수가 반대 방향으로 변할 때 (하나는 증가하고 다른 하나는 감소할 때) : 공분산 값은 음수(-)가 됩니다. -> 두 변수 사이에 음의 관계가 있음을 의미
- 두 변수가 서로 독립적으로 변할 때 (한 변수의 변화가 다른 변수의 변화와 관련이 없을 때): 공분산 값은 0에 가깝게 됩니다. -> 두 변수 사이에 선형적 관계가 없음을 의미

=> 공분산 값 자체는 변수들의 척도에 따라 달라질 수 있어, 관계의 강도를 직접 비교하기 어려움 (단순히 같이 증가하거나 같이 감소하면 공분산 값이 늘어나고, 그 반대면 음수가 되는 값)

- x1, x2의 공분산이 2000이고 x3, x4의 공분산이 400이라고 x3,x4 사이의 상관관계가 더 크다고 할 수는 없음.

=> 이런 이유로 관계의 강도를 표준화된 척도로 나타내기 위해 상관계수를 사용 (공분산을 변수 각각의 표준편차 값으로 나누는 normalilzation 정규화를 하여 상관성을 비교하기도 하지만, 이 역시도 절대적인 기준이 아니기에, 피어슨 상관계수를 많이 사용한다.)

> - **피어슨 상관계수**
- $r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}​$
- 함께 변하는 정도는 전체가 변하는 총량을 초과할 수 없기에 최댓값 = 1, 최솟값 = -1
- **산점도의 기울기는 상관계수와 관련이 없다.** 분산의 관게성이 같다면, 기울기가 크든 작든 상관계수는 같다.

=>> **상관계수가 높다 = X1이 움직일 때 X2가 많이 움직인다는 뜻이 아닌, X2를 예상할 수 있는 정확도, 즉 설명력이 높다는 것,**

![222](https://github.com/user-attachments/assets/f1861112-f434-4c9b-9d69-a21137e70092)

- 상관계수의 제곱 = 결정계수 = 총 변동 중에서 회귀선에 의해 설명되는 변동이 차지하는 비율
- 해당 독립 변수가 종속 변수의 변동을 설명하는 정도
- 두 변수의 선형 관계만을 측정할 수 있기에 2차 방정식 그래프와 같은 관계의 데이터는 상관계수가 매우 낮게 측정된다.

![333](https://github.com/user-attachments/assets/42140659-589c-4a3c-bc32-d92039fdcf9b)

**클러스터맵의 해석**
- 변수 간의 유사성 : 클러스터맵에서는 상관성이 높은 변수들이 클러스터링을 통해 가까이 위치합니다. 예를 들어, 변수 A와 변수 B가 상관성이 높다면, 클러스터맵에서는 A와 B가 서로 인접한 위치에 배치됩니다.
- 덴드로그램 : 클러스터맵은 히트맵 외에도 덴드로그램(dendrogram)을 포함하여 변수 간의 유사성을 트리 구조로 시각화합니다. 덴드로그램은 클러스터링의 계층 구조를 보여주며, 어떤 변수들이 함께 그룹화되는지를 나타냅니다.

> **요약**
- sns.clustermap을 사용하면 단순히 히트맵의 시각적 가독성을 높이는 것뿐만 아니라, 데이터 내 변수 간의 상관관계를 기반으로 한 실제 클러스터링을 통해 유사한 변수들을 그룹화하고, 이들을 인접한 위치에 배치하여 더욱 명확하게 상관관계를 파악할 수 있게 됩니다.

![444](https://github.com/user-attachments/assets/eacee72e-fa4c-42d3-b4d4-808a81e05709)

## **3. 시간 시각화**

- 시계열 형태로 표현되는 데이터의 시각화 => 시간 흐름에 따른 데이터의 변화 표현
- 전체적인 흐름 확인, 데이터의 트렌드, 노이즈 확인

> - 연속형 시간 시각화 => 선 그래프 : 시간 간격의 밀도가 높은 경우
- ex) 공정 데이터 or 연간 일별 판매량 데이터
- 추세선을 그려 데이터의 트렌드를 확인 -> 이동평균 방법 (25374 => (253),(537),(374)의 평균으로 게산)

> - 분절형 시간 시각화 : 막대그래프, 누적 막대그래프, 점그래프
- 시간의 밀도가 낮은 경우 활용
- 누적 막대 그래프로 한 시점에서의 다양한 세부 항목을 표현 가능


```python
df_line['Month'] = df_line['Sales'].rolling(window = 30).mean()

ax = df_line.plot(x = 'Date2', y = 'Sales', linewidth = "0.5")
df_line.plot(x='Date2', y='Month', color='#FF7F50', linewidth = "1", ax=ax)

# 날짜별로 매출액 편차가 커서 전체적인 추이를 보기위해서 rolling으로 이동 평균선을 만들어준다.
```

![output_15_1](https://github.com/user-attachments/assets/9c4bc799-a917-4d9b-a566-c4e878d1b637)

## 4. 비교 시각화

> - 그룹 별 요소가 많은 경우, **히트맵 차트**를 사용하여 각 그룹(행)을 기준으로 요소들(열)의 크기 비교 가능 & 각 요소를 기준으로 그룹들의 크기 비교 가능
- 각 그룹이 어떤 요소가 높고 낮은지 파악 & 요소 간의 관계도 파악 가능
- 행을 a 변수, 열을 b 변수, 셀의 값을 c변수로 설정할 수도 있음
- 차트의 열을 시간 흐름으로 설정하여 시간 시각화로도 활용 가능
- 분류 그룹이나 변수가 너무 많으면 혼란을 유발하기에 적정 수준 데이터 정제가 필요하다

> **방사형 차트** : 각 그룹의 여러 변수들의 값을 시각화 가능

> **평행 좌표 그래프 (전략 캔버스)**
- 변수별 값을 정규화(0 ~ 100 사이로 변환)하면 그래프를 보다 효과적으로 표현 가능


```python
# 팀 기준 평행 좌표 그래프 생성

fig,axes = plt.subplots()
plt.figure(figsize=(16,8)) # 그래프 크기 조정
parallel_coordinates(df3,'Tm',ax=axes, colormap='winter',linewidth = "0.5")
```

![output_17_1](https://github.com/user-attachments/assets/ed561184-9c3e-4274-92f2-b35470d8e078)

## 5. 분포 시각화
- 데이터가 처음 주어졌을 때, 변수들이 어떤 요소로 어느 정도의 비율로 구성되어 있는 지를 확인하는 단계

- **분포 시각화는 연속형과 같은 양적 척도인지, 명목형과 같은 질적 척도인지에 따라 구분해서 그린다.**
- 양적 척도(연속형)의 경우 : 막대그래프 or 선그래프로 분포 표현 / 히스토그램을 통해 분포 단순화
- 질적 척도(범주형)로 이루어진 변수는 구성이 단순한 경우 : 파이차트 or 도넛 차트
- **구성 요소가 복잡한 질적 척도를 표현할 때는 트리맵 차트 or 와플 차트를 이용하면 보다 효과적으로 표현 가능**




```python
# 남성 여성 구분하여 히스토그램 시각화

#  남성 여성 별도 데이터셋 생성
df1_1 = df[df['sex'].isin(['man'])]
df1_1 = df1_1[['height_cm']]
df1_2 = df[df['sex'].isin(['woman'])]
df1_2 = df1_2[['height_cm']]

# 10cm 단위로 남성, 여성 신장 히스토그램 시각화
plt.hist(df1_1, color = 'green', alpha = 0.2, bins = 10, label = 'MAN', density = True)
plt.hist(df1_2, color = 'red', alpha = 0.2, bins = 10, label = 'WOMAN', density = True)
plt.legend()
plt.show()
```

![output_19_0](https://github.com/user-attachments/assets/8b539c08-aca1-4d2a-aa19-188edfd1f8ab)

```python
fig = px.treemap(df3, path=['sex','country'], values='height_cm',
                 color='height_cm', color_continuous_scale='viridis')
fig.show()
```

![0400ae17-6fdf-4e38-a0c8-2ce6cdb432b5](https://github.com/user-attachments/assets/0d33a333-824a-42f7-a15f-6c3c6de63183)


## 6. 관계 시각화
- 산점도를 그릴 때는 극단치를 제거하고 그리는 것이 좋다. => 시각화의 효율이 떨어진다.
- 각각의 점에 투명도를 주어 점들의 밀도를 함께 표현 가능
- 값의 구간을 나누어 빈도에 따른 농도나 색상을 다르게 표현 가능

![555](https://github.com/user-attachments/assets/84d2858c-1b81-4f0e-9da0-b7b1e59f6be8)

## 7. 공간 시각화
-데이터가 지리적 위치와 관련되어 있는 경우, 실제 지도 위에 데이터를 표현
- 공간시각화는 위치 정보인 위도와 경도 데이터를 지도에 매핑하여 시각적으로 표현
- GOOGLE의 GEOMAP을 이용하여 지명만으로 시각화 가능
> 1. 도트맵 : 지리적 위치에 동일한 크기의 작은 점을 찍어서 해당 지역의 데이터 분포나 패턴을 표현하는 기법
> 2. 버블맵 : 데이터 값이 원의 크기로 표현되기에 코로플레스맵보다 비율을 비교하는 것이 효과적, 지나치게 큰 버블이 다른 지역을 침법하는 것에 주의
> 3. 코로플레스맵 or 링크맵 : 지도에 찍힌 점들을 곡선 or 직선으로 연결하여 지리적 관계를 표현 & 연속적인 연결을 통해 지도에 경로를 표현 가능
> 4. 시작점과 도착점이 함께 표현되는 커넥션 맵인 플로우맵
> 5. 각 지역의 면적을 데이터 값에 비레하도록 변형시켜 시각화하는 카토그램

![df95a5a4-d1aa-4ba4-a0be-372e94cfa040](https://github.com/user-attachments/assets/48dd4b6d-4a9c-41c4-93f4-ebac7d0d1d29)


## 8. 박스 플롯
- 하나의그림으로 양적 척도 데이터의 분포 및 편향성, 평균과 중앙값 등 다양한 수치를 보기 쉽게 정리해줌
- 두 변수의 값을 비교할 때 효과적

>- Q1 (제 1사분위수, 25th percentile): 데이터의 하위 25%를 나누는 값.
-Q3 (제 3사분위수, 75th percentile): 데이터의 상위 25%를 나누는 값.
-IQR (Interquartile Range, 사분위 범위): Q3 - Q1, 중간 50%의 데이터 범위.
- 최솟값 (Lower whisker): Q1 - 1.5 * IQR 이상인 가장 작은 값.
- 최댓값 (Upper whisker): Q3 + 1.5 * IQR 이하인 가장 큰 값.
- 중앙값 (Median): 데이터의 중앙값.
- 이상치 (Outliers): 최솟값과 최댓값 범위를 벗어나는 값. => 작은 원으로 표현

![image](https://github.com/user-attachments/assets/c7d03274-b1f2-4e57-95ef-08de02445599)


>- 박스 플롯(Box Plot)에서 말하는 "최솟값"은 원본 데이터의 최솟값과는 다릅니다. 박스 플롯의 최솟값과 최댓값은 이상치를 제외한 범위를 나타내기 위해 계산된 값입니다. 여기서 IQR(Interquartile Range, 사분위 범위)을 사용하여 이상치를 결정합니다.
- 즉, 박스 플롯의 최솟값은 원본 데이터의 최솟값이 아닌, 이상치를 제외한 후의 최솟값입니다.

- boxplot은 항상 데이터 분포도를 함께 떠올리는 습관이 필요하다.

> 위의 boxplot은 중앙값이 제 1사분위와 더 가깝기에 데이터가 오른쪽으로 치우쳐 있음을 알 수 있다.

> **데이터의 치우침 이해**
- 오른쪽으로 치우친 분포 (Right-Skewed Distribution): 데이터가 왼쪽에 집중되어 있고, 오른쪽 꼬리가 길게 늘어져 있는 분포입니다. 평균이 중앙값보다 큽니다.
- 왼쪽으로 치우친 분포 (Left-Skewed Distribution): 데이터가 오른쪽에 집중되어 있고, 왼쪽 꼬리가 길게 늘어져 있는 분포입니다. 평균이 중앙값보다 작습니다.


```python
import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터 생성
np.random.seed(10)
data = np.random.normal(loc=10, scale=2, size=100)
data_skewed_left = np.concatenate([data, np.random.normal(loc=5, scale=1, size=20)])
data_skewed_right = np.concatenate([data, np.random.normal(loc=15, scale=1, size=20)])

# 박스 플롯 생성
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].boxplot(data_skewed_left, vert=False)
axs[0].set_title('Left-Skewed Data')
axs[1].boxplot(data_skewed_right, vert=False)
axs[1].set_title('Right-Skewed Data')

plt.show()
```

![output_28_0](https://github.com/user-attachments/assets/a6472eb4-82a4-4c1e-9cea-53d5fa6e24a6)

```python
import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터 생성
np.random.seed(10)
data = np.random.normal(loc=10, scale=2, size=100)
data_skewed_left = np.concatenate([data, np.random.normal(loc=5, scale=1, size=20)])
data_skewed_right = np.concatenate([data, np.random.normal(loc=15, scale=1, size=20)])

# 히스토그램 생성
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
axs[0].hist(data_skewed_left, bins=20, color='skyblue', edgecolor='black')
axs[0].set_title('Left-Skewed Data')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

axs[1].hist(data_skewed_right, bins=20, color='lightcoral', edgecolor='black')
axs[1].set_title('Right-Skewed Data')
axs[1].set_xlabel('Value')

plt.tight_layout()
plt.show()
# 왼쪽으로 치우친 데이터 / 오른쪽으로 치우친 데이터
```
![output_29_0](https://github.com/user-attachments/assets/ba1542e6-2a8a-49e6-bcde-cd2dba99c8e9)
