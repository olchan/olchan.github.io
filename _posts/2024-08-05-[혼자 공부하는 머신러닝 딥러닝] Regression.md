## **주요 개념 정리**
#### **KNN regresion**
- 이웃한  sample의 target 값들의 평균

#### **1차원 리스트 형태의 data를 2차원으로 바꾸기**
- train_input = train_input.reshape(-1, 1)
- (n, ) : 1차원 배열 => (n,1) : 2차원 배열
- "-1"을 사용하면 배열의 전체 원소 개수를 매번 외우지 않아도 되므로 편리

### **결정 계수 R^2**
- 분류의 경우 : test sample을 정확하게 분류한 개수의 비율 : 정확도
- 회귀의 경우 : 결정계수 coeficient of determination R^2 로 회귀 모델 평가
- 1 - (타깃 - 예측)^2의 합 / (타깃 - 평균)^2의 합
- if 타깃이 평균 정도를 예측하는 수준이라면(기본 모델 정도라면), 즉 분자와 분모가 비슷해지면 R^2는 0에 가까워지고, 예측이 타깃에 아주 가까워지면 분자가 0에 가까워지기에, 1에 가까운 결정 계수 값을 가지게 된다.

-> 절대적인 평가값

-> 정량적인 평가값(예측이 얼마나 벗어났는가!) : 절댓값 오차

#### 과대적합 / 과소적합
- 훈련 세트와 테스트 세트의 점수 차이가 크면 좋지 않다.
- 일반적으로 훈련 세트의 점수가 테스트 세트보다 조금 더 높다.
- 테스트 세트의 점수가 너무 낮으면 과대 적합 (over fitting)
- 테스트 점수가 너무 높거나 두 점수가 모두 낮으면 과소 적합 (under fitting)

> 사이킷 런에 사용하는 훈련 세트는 2차원 배열이어야 한다.
- train_input = train_input.reshape(-1, 1)




```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

# KNN 회귀 모델 훈련
knr.fit(train_input, train_target)
```


```python
knr.score(test_input, test_target)

# 분류의 경우 : 정확도 (sample을 정확하게 분류한 개수의 비율)
# 회귀의 경우 : 결정계수 coefficient of determination
```




    0.992809406101064

![image](https://github.com/user-attachments/assets/14fb8e12-2125-41a0-b084-dc2d10f8f350)

- SST : 관측값에서 관측값의 평균을 뺸 결과의 총합 => data의 변동성을 의미
- SSE : 추청 값에서 관측값의 평균을 뺀 결과의 총합
- SSR : 관측값에서 추정값을 뺸 값 => 잔차의 총합 => 회귀 모델이 설명하지 못한 변동성
- SST = SSR + SSE
- SSE가 커진다는 것은 SSR이 작아진다는 것이고, SSE가 작아지면 설명 불가능한 변동이 작아지는 거니까,우리가 추정한 모형을 바탕으로 반응변수 Y를 보다 잘 예측할 수 있게 된다는 것
- 결정계수 : 회귀 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 척도

> - 결정 계수 $𝑅^2$는 독립 변수의 개수가 늘어나면 일반적으로 증가하는 경향이 있습니다.
- 이는 모델에 새로운 독립 변수를 추가하면 모델이 훈련 데이터에 더 잘 맞출 수 있는 더 많은 자유도를 가지게 되기 때문입니다.
- 하지만 이러한 증가가 모델의 설명력(즉, 일반화 성능)이 실제로 높아지는 것을 의미하는 것은 아닙니다.

![image](https://github.com/user-attachments/assets/a5aa6602-9997-471a-bdaf-895a4783a86f)


```python
print(knr.score(train_input, train_target)) # -> 과소 적합
```

    0.9698823289099254
    

- 과소 적합을 해결 -> 모델을 더 복잡하게!
- knn 알고리즘에서는 k를 줄인다.
- 이웃의 개수를 늘리면 데이터의 전반적인 패턴을 학습하고, 이웃의 개수를 줄이면 데이터의 지엽적인 패턴을 학습


```python
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
```

    0.9804899950518966
    

## **주요 개념 정리**
- KNN 알고리즘의 한계
  - 아무리 멀리 떨어져 있는 data라도 무조건 가장 가까운 샘플의 타깃을 평균하여 예측하기에,훈련 data set 범위 밖의 data에 대해서는 입력값에 상관없이 같은 값의 예측값을 반환하여 예측이 불가하다.

### 선형 회귀 : data를 설명하는 최적의 직선 찾기


```python
# LinearRegression 클래스가 찾은 a와 b는 lr객체의 coef_와 intercept_ 속성에 저장되어 있다.
print(lr.coef_, lr.intercept_)
```
- in 머신러닝
- **기울기** : coefficient 계수 or weight 가중치 라고 부름
- coef_와 intercept_를 머신러닝 알고리즘이 찾은 값이라는 의미로 model parameter라고 부름 => **모델 기반 학습**
- KNN 알고리즘 : 모델 파라미터가 없고, 훈련 세트를 저장하는 것이 훈련의 전부였음 => **사례 기반 학습**



```python
# 기본적인 직선 그래프 그리기
plt.plot(x,y)
# 입력값이 하나일 경우, y 값이라고 인식
# 선을 그릴때, 다양한 선의 종류를 선택할 수 있다. 디폴트가 직선이고, 점으로 표현하는 마커나 점선등을 선택 가능
```

### 다항 회귀 : data를 설명하는 최적의 곡선 찾기

- **2차 방정식 그래프를 찾기 위해 훈련 세트에 제곱항을 추가했지만, 타깃값은 그대로 사용한다.**
- **목표하는 값은 어떤 그래프를 훈련하든지 바꿀 필요가 없다.**

> - coef_ : 특성에 대한 계수
- intercept_ : 절편이 포함


#### **사이킷런의 PolynomialFeatures 클래스**
- 사이킷런에서 특성features을 만들거나 전처리하기 위한 모델 클래스 : Transformer
- fit(), transform() 메서드
- transformer를 fit하면 만들 특성의 조합을 준비함
- transform 메서드 : **각 특성을 제곱한 항, 특성끼리 서로 곱한 항, 1**
- 1: 선형 방정식의 절편은 항상 값이 1인 특성과 곱해지는 계수라고 생각 => **사이킷런의 선형 모델은 자동으로 절편을 추가해주기에, include_bias = False 로 정해서 특성 변환**
- **PolynomialFeatures 클래스의 degree 변수 사용** => 필요한 고차항의 최대 차수를 지정 가능
- But, 특성의 개수를 크게 늘리면 선형 모델은 훈련 set에 대해서만 거의 완벽하게 학습 가능 => 과대 적합

### **규제 regularization**
- 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
- 모델이 훈련 set에 과대적합되지 않도록 만드는 것
- 선형 회귀 모델의 경우 특성에 곱해지는 계수(기울기)의 크기를 작게 만드는 일

#### StandardScaler() 클래스 : 표준화를 하는 사이킷런 클래스

## **릿지 Ridge 라쏘 Lasso**
- 릿지 : 계수를 제곱한 값을 기준으로 규제를 적용
- 라쏘 : 계수의 절댓값을 기준으로 규제를 적용

- 일반적으로 라쏘를 선호
- 두 알고리즘 모두 계수의 크기를 줄이지만 라쏘는 아예 0으로 만들 수도 있음 => **유용한 특성을 골라내는 용도로도 사용 가능**


- 많은 특성을 사용했음에도 불구하고 훈련 세트에 너무 과대적합되지 않아 테스트 세트에서도 좋은 성능을 냄
- 릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절 가능
- 모델 객체를 만들 때 alpha 매개변수로 규제의 강도를 조절
- alpha값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도
- alpha 값이 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 유사해지므로 과대적합 가능성이 증가

#### alpha 값 :  머신러닝 모델이 학습하는 parameter(변수)가 아닌 사람이 직접 입력해야 하는 파라미터 => hyperparameter

> **적절한 alpha 값은 R^2 그래프를 통해 결정 가능**
- 훈련 set와 테스트 set의 점수가 가장 가까운 지점이 최적의 alpha 값이 됨

![image](https://github.com/user-attachments/assets/466f6828-5f97-44dc-a368-f006dfb9c6f5)

![image](https://github.com/user-attachments/assets/8aa1fbec-62a5-4600-9486-6c36ade17d31)

```python
point = np.arange(15,50)

plt.scatter(train_input, train_target)

plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

plt.scatter(50,1574, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![output_9_0](https://github.com/user-attachments/assets/38c81a05-10c6-4304-97c4-cb0835335a0e)


> 2차 방정식을 포함한 다항 회귀(polynomial regression)도 선형 회귀
-  회귀 분석에서 "선형"이라는 용어가 모델의 형태가 아닌, 파라미터에 대해 선형적임을 의미하기 때문입니다. 즉, 선형 회귀는 종속 변수 𝑦가 독립 변수 x의 함수로 표현될 때, 그 함수의 계수(파라미터)가 선형적으로 나타나는 경우를 말합니다.
- 비선형 회귀의 경우 : 종속 변수 y가 독립 변수 x의 비선형 함수로 표현되는 경우

## **주요 개념 정리**

#### 다중 회귀 : 여러 개의 특성을 사용한 선형 회귀
- 특성이 1개면 선을 학습
- 특성이 2개면 면을 학습
- 특성이 3개 이상이면 그릴 수 없는 공간에서 표현 -> 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현 가능

#### **특성 공학 feature engineering** : 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업



```python
lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

    0.9888067471131867
    0.9824470598706695
    


```python
print(np.sum(lasso.coef_ == 0))
# 라쏘 모델은 55개의 feature들 중 15개의 특성만을 사용
```

    40
    
