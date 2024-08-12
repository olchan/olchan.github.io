## **결정 트리 - 주요 개념 정리**
- 누락된 값은 data를 버리거나, 평균값, 중앙값으로 대체한다.
- **훈련 세트의 평균값으로 테스트 세트의 누락된 값을 채워야 한다.** (무조건 훈련 set의 통계값으로 테스트 세트를 변환한다!)
--------
- Logistic regression 모델은 학습의 결과를 설명하기 어려움 => 이유를 설명하기 쉬운 결정 트리 모델

### **Decision Tree 결정 트리**
- 최종 분류된 sample의 class 예측 : 리프 노드에서 가장 많은 클래스를 예측 클래스로 함
- in Decsion Tree regressor : 리프 노드에 도달한 샘플의 타깃을 평균하여 예측값으로 사용

#### plot_tree() 함수를 사용하여 결정 트리를 해석하기
- max_depth 매개변수를 통해 트리의 깊이를 제한 가능
![image](https://github.com/user-attachments/assets/66754172-14a1-4f59-82bd-f429f16e1ede)

>1. 테스트 조건
2. 불순도 - 지니 불순도
3. 총 샘플의 수
4. 클래스별 샘플의 수

#### 지니 불순도 = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)
- Decision Tree 모델은 부모 노드와 자식 노드의 불순도 차이가 가능한 크도록 트리를 성장시킴

- **부모 노드와 자식 노드의 불순도 차이 계산** => 정보 이득 information gain

> 1. 자식 노드의 불순도를 샘플 개수에 비례하여 모두 더함
2. 부모 노드의 불순도에서 1번 값을 뺀다.

#### 엔트로피 불순도

![image](https://github.com/user-attachments/assets/959a2307-9fe9-4653-8158-fda0a66d12f1)


- 새로운 sample에 대해 예측할 때에는 노드의 질문에 따라 트리를 이동한다. 그리고 마지막에 도달한 노드의 클래스 비율을 보고 예측을 한다.

### 가지 치기
- 가지 치기를 하지 않으면 트리가 끝까지 자라난다. => 훈련 set에는 잘 맞지만 test set에서는 성능이 구현되지 않는다.
- 자라나는 트리의 최대 깊이를 지정

**불순도는 클래스별 비율을 가지고 계산했고, 불순도를 기준으로 sample을 나누기에, sample을 어떤 클래스 비율로 나누는 지 계산할 때 특성값의 스케일이 계산에 영향을 미치지 않는다. => Tree 모델을 훈련시킬 때에는 input값을 표준화 전처리를 할 필요가 없다.**

## **교차 검증과 그리드 서치 - 주요 개념 정리**
- 결정 트리의 다양한 매개변수, 즉 hyper parameter를 자동으로 찾기 위한 방법
- hyper parameter를 결정하려면 많이 시도해보아야 하는데, 테스트 set을 사용해 자꾸 성능을 확인하다 보면, 점점 모델이 test set에 맞게 변화하게 됨
- test set으로 일반화 성능을 올바르게 예측하기 위해서 가능한 한 test set을 사용하지 말고 모델을 만들고 나서 마지막에 딱 한 번만 사용하는 것이 좋음

#### **검증 세트 validation set**
- 훈련 set와 test set으로 나눈 data에서 훈련 set을 또 다시 validation set과 훈련 set으로 나눈다.
1. 훈련 set에서 모델을 훈련하고 검증 세트로 모델을 평가
2. 테스트하고 싶은 매개변수를 바꿔가며 가장 성능이 좋은 매개변수로 만들어진 모델을 결정
3. 이 매개변수들을 사용해 훈련 set과 validation set을 합친 전체 훈련 data로 모델을 다시 학습
4. 마지막으로, 테스트 set에서 최종 점수를 평가



```python
# train_test split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state= 42)

# train_test_split() 함수를 2번 적용해서 validation set split
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state = 42)
```

### **교차 검증 Cross Validation**
- validation set을 만들게 되면 훈련 set이 줄어들게 됨

1. 교차 검증 : 검증 set을 떼어 내어 평가하는 과정을 여러번 반복
2. 이 점수를 평균하여 최종 검증 점수를 얻음

- 안정적인 검증 점수를 얻고 훈련에 더 많은 data를 사용 가능


```python
# sklearn의 cross_validate() : 교차 검증 함수
# 1. 평가할 모델 객체를 1번째 매개변수로 전달
# 2. 직접 검증 set을 분리하지 않고, 훈련 set 전체를 cross_validate 함수에 전달
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
# fit_time, score_time : 각각 모델을 훈련하는 시간, 검증하는 시간
# 각 key 마다 5개의 숫자가 담겨 있음 = > cross_validate 함수는 기본적으로 5 - fold 교차 검증을 수행
# cv 매개변수를 통해 폴드 수를 지정 가능
# test_score 키 : validation set을 통한 모델 점수 => 최종 점수 : test_score 키에 담긴 5개의 점수를 평균하여 얻음
```
회귀 모델 : KFold 분할기를 기본으로 사용
분류 모델 : 타깃 class를 골고루 나누는 stratifiedKFold를 기본으로 사용
#### **splitter 분할기**

![image](https://github.com/user-attachments/assets/be646cbf-ed62-4f2f-9e4b-56be0d54cd00)

![image](https://github.com/user-attachments/assets/259eea98-7574-45ca-b99e-c7185387b7f8)

### Hyper parameter tuning 하이퍼 파라미터 튜닝
- 머신러닝 모델이 학습하는 파라미터 : model parameter
- 모델이 학습할 수 없고, 사용자가 지정해야만 하는 parameter : hyper parmeter
- 사이킷런과 같은 머신러닝 라이브러리를 사용할 때 이런 하이퍼 파라미터는 모두 클래스나 메서드의 매개변수로 표현

### Auto ML : 사람의 개입 없이 하이퍼 파라미터를 자동으로 수행하는 기술

- 1개의 하이퍼 파라미터의 최적값을 찾은 후, 그 값을 고정하고 다른 하이퍼 파라미터의 최적값을 찾는 것은 불가능
- 하나의 값이 달라지면 다른 매개변수의 최적값도 달라지기에 여러 매개변수를 동시에 바꿔가면서 최적의 값을 찾아야 함

### **GridSearchCV 그리드 서치**
- 하이퍼 파라미터 탐색 & 교차 검증을 한 번에 수행 => 별도로 cross_validate() 함수를 호출하지 않아도 됨

```python
# n = -1 : 병렬 실행에 사용할 cpu 코어 수를 지정
# default 값 : 1
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)
```
1. 탐색할 매개변수를 dictionary 타입으로 입력
2. 훈련 set에서 grid search를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾기 => 이 조합은 그리드 서치 객체에 저장됨
3. **그리드 서치는 최상의 매개변수에서 전체 훈련 set를 사용해 최종 모델을 훈련함 => 이 모델은 'best_estimator_' 객체에 저장됨**

- np.arange(a,b,c) : a~b-1 까지 c를 계속 더한 배열
- range(a,b,c) : a ~ b-1 까지 c를 계속 더한 정수 배열

### **Random Search 랜덤 서치**
- **매개변수 값의 목록을 전달하는 것이 아닌, 매개변수를 sampling할 수 있는 확률 분포 객체를 전달**
- **싸이파이 Scipy 라이브러리를 활용**
- uniform & randint 클래스 => 주어진 범위에서 고르게 값을 뽑음

```python
# n_iters : sampling 횟수
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42), params, n_iter = 100, n_jobs = -1, random_state = 42)
gs.fit(train_input, train_target)
```
- 매개 변수 값이 수치형, 특히 연속적인 실숫값이라면 싸이파이의 확률 분포 객체를 전달하여 특정 범위 내에서 지정된 횟수만큼 매개변수 후보 값을 샘플링하여 교차 검증을 시도 가능

- 한정된 자원을 최대한 활용하여 효율적으로 하이퍼 파라미터 공간을 탐색할 수 있는 아주 좋은 도구

=>  수동으로 매개변수를 바꾸지 않고, 그리드 서치나 랜덤 서치를 사용

## **트리의 앙상블 - 주요 개념 정리**
#### **정형 data VS 비정형 data**
- 정형 data : csv, database, excel에 저장하기 쉬움
- 비정형 data : 글, 사진, 음악.. -> NoSQL database는 excel이나 csv에 담기 어려운 text, json data를 저장하는데, 용이하다.
- 지금까지 학습한 머신러닝 알고리즘은 정형 data에 잘 맞음
- 특히 **정형 data를 다루는 데 가장 뛰어난 알고리즘 : 앙상블 학습**
- 비정형 data는 규칙성을 찾기 어려움 => 신경망 알고리즘을 사용해 학습

#### **앙상블 학습 ensemble learning**
- decision tree를 기반으로 만들어진 알고리즘
- 여러 개의 결정 트리(Decision Tree)를 결합하여 하나의 결정 트리보다 더 좋은 성능을 내는 머신러닝 기법
-  여러 개의 약 분류기 (Weak Classifier)를 결합하여 강 분류기(Strong Classifier)를 만드는 것

### **배깅(Bagging) & 부스팅(Boosting)**
**배깅(Bagging)**
- Bootstrap Aggregation의 약자
- 샘플을 여러 번 뽑아(Bootstrap) 각 모델을 학습시켜 결과물을 집계(Aggregration)하는 방법

> 우선, 데이터로부터 부트스트랩을 합니다. (복원 랜덤 샘플링) 부트스트랩한 데이터로 모델을 학습시킵니다. 그리고 학습된 모델의 결과를 집계하여 최종 결과 값을 구합니다.

> Categorical Data는 투표 방식(Votinig)으로 결과를 집계하며, Continuous Data는 평균으로 집계합니다.

> Categorical Data일 때, 투표 방식으로 한다는 것은 전체 모델에서 예측한 값 중 가장 많은 값을 최종 예측값으로 선정한다는 것입니다. 6개의 결정 트리 모델이 있다고 합시다. 4개는 A로 예측했고, 2개는 B로 예측했다면 투표에 의해 4개의 모델이 선택한 A를 최종 결과로 예측한다는 것입니다.

> Continuous Data일 때, 평균으로 집계한다는 것은 말 그대로 각각의 결정 트리 모델이 예측한 값에 평균을 취해 최종 Bagging Model의 예측값을 결정한다는 것입니다.

**부스팅(Boosting)**

> 부스팅은 가중치를 활용하여 약 분류기를 강 분류기로 만드는 방법입니다. 배깅은 Deicison Tree1과 Decision Tree2가 서로 독립적으로 결과를 예측합니다. 여러 개의 독립적인 결정 트리가 각각 값을 예측한 뒤, 그 결과 값을 집계해 최종 결과 값을 예측하는 방식입니다. 하지만 부스팅은 모델 간 팀워크가 이루어집니다. 처음 모델이 예측을 하면 그 예측 결과에 따라 데이터에 가중치가 부여되고, 부여된 가중치가 다음 모델에 영향을 줍니다. 잘못 분류된 데이터에 집중하여 새로운 분류 규칙을 만드는 단계를 반복합니다.

![image](https://github.com/user-attachments/assets/bd37eb5e-c7cd-4808-acdb-9a25029a9d1a)

> **배깅은 병렬로 학습하는 반면, 부스팅은 순차적으로 학습합니다. 한번 학습이 끝난 후 결과에 따라 가중치를 부여합니다. 그렇게 부여된 가중치가 다음 모델의 결과 예측에 영향을 줍니다.**

> **오답에 대해서는 높은 가중치를 부여하고, 정답에 대해서는 낮은 가중치를 부여합니다. 따라서 오답을 정답으로 맞추기 위해 오답에 더 집중할 수 있게 되는 것입니다.**

> **부스팅은 배깅에 비해 error가 적습니다. 즉, 성능이 좋습니다. 하지만 속도가 느리고 오버 피팅이 될 가능성이 있습니다. 그렇다면 실제 사용할 때는 배깅과 부스팅 중 어떤 것을 선택해야 할까요? 상황에 따라 다르다고 할 수 있습니다. 개별 결정 트리의 낮은 성능이 문제라면 부스팅이 적합하고, 오버 피팅이 문제라면 배깅이 적합합니다.**

### **RandomForest** : 앙상블 학습의 대표적인 알고리즘
1. 각각의 트리를 훈련시키기 위한 data를 랜덤하게 생성 -> 입력한 훈련 data에서 랜덤하게 sample을 복원 추출하여 훈련 data를 만든다.
- bootstrap sample : 중복을 허용하여 random으로 뽑은 sample
- 훈련 set의 크기와 같은 size로 만듦
- 각 노드를 분할할 때, 전체 특성 중 일부 특성을 무작위로 고른 다음 이 중에서 최선의 분할을 찾음 => RandomForestClassifier : 전체 특성 개수의 제곱근만큼의 특성을 선택
- 모든 노드를 만들 때 이 과정을 반복
- 회귀 모델인 RandomForestRegressor : 전체 특성을 사용

2. 사이킷런의 random forest는 기본적으로 100개의 결정 트리를 이런 방식으로 훈련
- 분류 : 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측값으로 결정
- 회귀 : 단순히 각 트리의 예측값을 평균

=> 랜덤하게 선택한 샘플과 특성을 사용하기에, 훈련 세트에 과대적합되는 것을 방지, 검증 set와 테스트 set에서 안정적인 성능을 얻을 수 있음

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
# 교차 검증을 통해 훈련 score (Cross_Validate() 함수의 'train_score')와 validation score ('test_score')를 알아냄
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)
# K가 5일 때, train_score는 validation fold가 아닌 나머지 4개의 train_fold를 학습한 모델의 점수 => train_fold로 학습했다고 해서 모델이 해당 data set을 100% 설명하는 것이 아님
# test_score : validataion_fold로 모델을 평가한 점수
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# cross_validate()로 나뉘어진 train과 validation set 각각이 RandomForest모델에 학습됨
```
- rf.oob_score_ : bootstrap sample 생성 과정에서 select되지 못한 OOB (out of bag) sample로 훈련한 decision tree 를 평가한 score


### **Extra Tree**
- bootstrap sample을 사용 x
- 노드를 분할 시, splitter 매개변수를 best가 아닌 random으로 설정한 것과 같이, 가장 좋은 분할이 아닌 무작위로 분할을 함
- 특성을 무작위로 분할하면 성능이 낮아지겠지만, 많은 트리를 앙상블하기 때문에 과대적합을 방지하고 검증 set의 점수를 높임
- random하게 노드를 분할하기에, 빠른 계산 속도가 장점

---

> **Gradient Descent** : https://angeloyeo.github.io/2020/08/16/gradient_descent.html

=> seta를 계속해서 수정해나가는 방식, 손실함수의 global minimum을 향해 점진적으로 내려가는 방식

## **Gradinet Boosting 그래디언트 부스팅**
-  결정 트리를 계속 추가하면서 손실함수의 가장 낮은 곳을 찾아 이동
- 깊이가 얕은 tree를 사용하여, 손실함수의 낮은 곳으로 천천히 조금씩 이동
- 학습률 매개변수로 속도를 조절

- 부스팅의 대표적인 모델은 AdaBoost, Gradient Boost등이 있음
- Gradient Boost의 변형 모델로는 XGBoost, LightGBM, CatBoost가 있음

### Boosting을 하는 방식에도 2가지가 있다.

1. Ada Boost와 같이 중요한 data(일반적으로 모델이 틀리게 예측한 data)에 대해 weight를 주는 방식

2. GBDT(Gradient Boosting Decision Tree)와 같이 loss function 처럼 정답지와 오답지간의 차이를 반복적으로 training하는 방식 => gradient를 이용해서 모델을 개선하는 방식 (XGBOOST, Light GBM)

![image](https://github.com/user-attachments/assets/55a00b1a-b282-4bb9-8358-cdf779e207c0)