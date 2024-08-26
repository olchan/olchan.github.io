> 군집 알고리즘

- clustering 군집 : 비슷한 샘플끼리 그룹으로 모으는 작업
- cluster : 군집 알고리즘에서 만든 군집
#### **비지도 학습**
- 사진의 pixel 값을 모두 평균 내어 pixel값의 평균이 비슷한 과일끼리 묶어보면 과일이 종류별로 묶이지 않을까?
- 과일이 전체 3가지 종류인 것을 인지한 상태에서 진행한다고 가정

```python
plt.imshow(fruits[0], cmap = 'gray_r')
```
> cmap = 'gray' : 컴퓨터가 분석을 편하게 하기 위함
- 보통의 흑백 이미지는 바탕이 밝고 물체가 짙은 색
- 분석을 위해 cmap 매개변수를 gray로 설정해주어 물체를 밝게 하고 배경을 어둡게 해서 물체가 있는 pixel 값이 높은 값을 가지도록 변환시켜준다.
- 밝을수록 높은 픽셀값을 가짐 어두우면 0
- **알고리즘이 어떤 출력을 만들기 위해 곱셈, 덧셈을 하는데, 픽셀값이 0이면 출력도 0이되어 의미가 없다. 픽셀값이 높아야 출력도 커지기에 의미를 부여하기 좋다.**
- cmap = 'gray_r' : 사람이 주로 보는 흑백 이미지

#### Matplotlib의 subplot()


```python
fig, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap = 'gray_r')
axs[1].imshow(fruits[200], cmap = 'gray_r')
plt.show()
```

> K-Means Clustering

- 비지도 학습은 전체 data가 몇개의 cluster로 분류되는 지, 해당 cluster가 나타내는 정체가 무엇인지 알 수 없음
- K - means 알고리즘 : 평균값을 자동으로 찾아줌
- 알고리즘으로 찾은 평균값 : 센트로이드, 클러스터 중심이라고 부름

> 1. 무작위로 k개의 클러스터 중심 정하기
2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복

> - np.ceil : 인수로 받은 숫자를 올림하여 반환
- np.round() : 반올림
- np.floor : input값을 내림하여 정수로 반환

- Boolean indexing
> draw_fruits(fruits[km.labels_ == 0])
- km.labels_ 배열에서 값이 0인 위치는 true, 그 외는 false가 됨
- numpy 배열에 불리언 인덱싱을 적용하면 True인 위치의 원소만 모두 추출

> kkk[a] VS kkk[a:a+1]
- index가 a인 요소만 추출 VS 슬라이싱으로 list의 일부를 추출
- 1개의 인덱스만 뽑고 싶더라도 슬라이싱으로 추출해야, 차원에 변형 없이 추출됨

### 최적의 K 구하기 : Elbow 방법
- 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법
- 클러스터 개수를 증가시키면서 이너셔를 그래프로 그리면 감소하는 속도가 꺾이는 지점이 있는데, 이 지점부터는 클러스터 개수를 늘려도 클러스터에 잘 밀집된 정도가 잘 개선되지 않음

#### inertia 이너셔
- k 평균 알고리즘은 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 잴 수 있는데, 이 거리의 제곱 합을 이너셔라고 부른다.
- 이너셔 : sample이 얼마나 클러스터의 중심에 가깝게 분포하고 있는 지를 나타내는 값


> 주성분 분석

## **주요 개념 정리**
### **Principal Component Analisys CPA 주성분 분석**
- 이전 장의 과일 data :  10000개의 pixel(100 x 100 size)이 있기 때문에 10000개의 특성이 있다고 할 수 있음
- 차원을 줄이면 저장 공간을 줄이고, 계산량을 줄일 수 있음

> - 2차원 배열에서는 행과 열이 차원
- 1차원 배열에서는 원소의 개수 자체가 차원

- 데이터에 있는 분산이 큰 방향의 벡터를 찾음
- 전체 데이터를 가장 잘 설명하는 벡터

- 주성분은 최대 원본 특성의 개수만틈 찾을 수 있음

```python
# n_components = 50 => 주성분의 개수 지정
# 전체 data shape : (300, 10000) =>  주성분 50개를 구하고 transform하면 (300,50)으로 전체 feature가 10000 -> 50개로 줄어듬
# data shape에서 첫번째 300은 batch dimension / not a feature
from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```
---
#### 원본 데이터 재구성
PCA로 줄인 feature는 다시 원본 data로 복구할 수 있다.

```python
# 원본 data 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
```
---
#### 설명된 분산
- 주성분이 원본 data를 얼마나 잘 나타내는지(잘 설명하는지) 기록한 값 : 설명된 분산
- PCA 클래스의 "explained_variance_ratio_"에 각 주성분의 설명된 분산 비율이 기록되어 있음

> 주성분의 개수 대신 **n_components = 0.n** 으로 설명된 분산의 비율을 입력해줄 수도 있음

