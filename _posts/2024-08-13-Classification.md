## **로지스틱 회귀**

- k-최근접 이웃 분류기 : KNN Classification
### 다중 분류 multi-class classification
- 이진 분류와 모델을 만들고 훈련하는 방식은 동일
- 이진 분류에서는 양성 클래스와 음성 클래스를 각각 1과 0으로 지정하여 타깃 data를 만들었었음
- 다중 분류에서도 타깃값을 숫자로 바꾸어 입력할 수 있지만 사이킷런에서는 문자열로 된 타깃값을 그대로 사용할 수 있음
- 이때, 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨진다. => pd.unique(fish['species'])로 출력했던 순서와 차이가 생김

- **KNeighborsClassifier에서 정렬된 타깃값은 classes_ 속성에 저장되어 있음**
- predict_proba() 메서드로 클래스별 확률값을 반환
- kneighbors() 메서드의 입력은 항상 2차원 배열이어야만 함
- 이를 위해, numpy 배열의 슬라이싱 연산자 사용 => 슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열이 만들어짐

## Logistic Regression
- 선형 회귀와 동일하게 선형 방정식을 학습함
- z = (a) x (feature1) + (b) x (feature2) + ... + f
- 위의 z 값을 **sigmoid function 시그 모이드 함수**에 넣는다. => 이진 분류에 사용


## **Gradient Descent 경사 하강법**
- 경사 하강법(Gradient Descent): 손실 함수의 기울기(경사)를 따라 최소값을 찾아가는 최적화 알고리즘입니다. 최적화의 목표는 손실 함수를 최소화하는 것이며, SGD는 이를 위해 매 반복에서 현재 위치에서의 기울기(경사)를 이용하여 파라미터를 업데이트합니다.

- 회귀 문제에서 경사 하강법을 사용할 때, 최적화하려는 손실 함수의 변수는 주로 모델의 파라미터(가중치와 편향) => θ (seta)

- 경사 하강법은 이 'θ' 값을 조정하여 손실 함수를 최소화하는 것이 목표입니다.
- 최적화는 주어진 데이터에 대한 예측 오차를 최소화하도록 모델 파라미터를 조정하는 과정을 의미합니다. 이를 통해 모델은 데이터에 가장 잘 적합되는 파라미터를 찾아가게 됩니다.
-  단계에서 계산된 gradient는 해당 데이터 포인트에 대한 손실 함수의 기울기를 의미
- local minimum은 각 단계마다 바뀌게 됨
- 전체 데이터셋에 대한 손실 함수의 최적값을 찾는 것이 아니라 각 단계에서의 지역적인 최적값을 찾는 것이기 때문에 SGD는 궁극적으로 더 빠르게 수렴할 수 있습니다. 그러나 이러한 특성 때문에 SGD는 노이즈가 많은 경우에 전체 데이터셋을 기반으로 한 Batch Gradient Descent보다 더 불안정할 수 있습니다.

점진적인 학습
1. 기존의 훈련 data에 새로운 data를 추가하여 모델을 다시 훈련 시키는 방법 -> 시간이 지날수록 data 양이 늘어남
2. 새로운 data를 추가할 때 이전 데이터를 버림으로써 훈련 data 크기를 일정하게 유지하는 방법 => 버린 data에 중요한 정보가 있을 수도 있음

=> 앞서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 방법

#### **점진적 학습 (= 온라인 학습)** = 산(경사)을 내려가는 것과 유사
- **확률적 경사 하강법** **Stochastic Gradient Descent**
- 확률적 : 무작위, 랜덤하게

=> 전체 샘플을 사용하지 않고 딱 하나의 샘플을 훈련 set에서 랜덤하게 고르는 방법
- 훈련 set에서 랜덤하게 하나의 sample을 선택하여 가파른 경사를 조금 내려간다.
- 그다음, 훈련 set에서 랜덤하게 또 다른 샘플을 하나 선택하여 경사를 조금 내려간다.
- 이 과정을 전체 sample을 모두 사용할 때까지 반복한다.
- if 모든 sample을 다 사용했는데도, 산을 다 내려오지 못한다면, 처음부터 다시 시작

#### **에포크 epoch** : 확률적 경사 하강법에서 훈련 세트를 한 번 모두 사용하는 과정

-> **훈련 data가 모두 준비되어 있지 않고 매일매일 업데이트되어도 학습을 이어나갈 수 있기에, 다시 산꼭대기에서부터 시작할 필요가 없다.**

> in 신경망, 많은 data를 사용하기에, 한 번에 모든 data를 사용하기에 어렵고, 모델이 매우 복잡하기 때문에 수학적인 방법으로 해답을 얻기 어려움
=> 확률적 경사 하강법 이나 미니배치 경사 하강법을 사용함

### **Loss function 손실 함수 == Cost Function 비용 함수**
- 손실함수의 값은 작을 수록 모델의 성능이 좋다는 것인데, 어떤 값이 최솟값인지는 알 수 없다.
- 가능한 많이 찾아보고 만족할만한 수준이라면 산을 다 내려왔다고
인정해야함
- but, 자주 다루는 문제에 필요한 손실 함수는 이미 정의되어 있기에 괜찮다!
- 손실 함수는 미분 가능해야 함!!

#### **이진 분류 : 로지스틱 손실 함수, 이진 크로스 엔트로피 손실 함수**
- 타깃이 양성 클래스(1)일때 : -log(p)
- 타깃이 음성 클래스(0)일때 : -log(1-p)

#### **다중 분류 : 크로스 엔트로피 손실 함수**

- 회귀 : MSE 평균 제곱 오차 => 타깃에서 예측을 뺸 값을 제곱한 다음 모든 sample에 평균한 값

#### SGDClassifier
- tol 매개변수 : 반복을 멈출 조건
- n_iter_no_change 매개변수에서 지정한 epoch 동안 손실이 tol만틈 줄어들지 않으면 알고리즘이 중단

Scikit-learn의 SGDClassifier를 OvR(One-vs-Rest) 방식으로 사용할 때, 각 클래스에 대한 이진 분류자를 생성하여 하나의 클래스를 양성 클래스로 다른 클래스들을 음성 클래스로 분류합니다. 이진 분류자마다 클래스에 대한 판별 경계를 설정합니다.

기본적으로 SGDClassifier에서 어떤 클래스가 양성 클래스로 취급될지를 결정하는 특별한 파라미터나 코드는 제공되지 않습니다. 이것은 OvR 분류 방식의 핵심 개념입니다. 어떤 클래스가 양성 클래스로 간주될지는 내부적으로 알고리즘에 의해 자동으로 처리되며 각 클래스에 대해 독립적인 이진 분류자가 생성됩니다.

만약 특정 클래스를 양성 클래스로 지정하려면, 다음과 같이 SGDClassifier의 partial_fit 메서드를 사용하여 해당 클래스와 다른 모든 클래스에 대해 이진 분류자를 독립적으로 학습시킬 수 있습니다. 아래의 예제는 레이블이 1인 클래스를 양성 클래스로 지정하고 다른 클래스들과의 이진 분류자를 학습하는 방법을 보여줍니다:


```python
from sklearn.linear_model import SGDClassifier

# SGDClassifier 인스턴스 생성
classifier = SGDClassifier(loss='log', random_state=42)

# 클래스 1을 양성 클래스로 설정
positive_class = 1

# 데이터와 해당 클래스의 레이블을 준비
X, y = your_data, your_labels

# 클래스 1에 대한 이진 분류자 학습
classifier.partial_fit(X, y == positive_class, classes=[0, 1])
```

여기서 your_data는 데이터, your_labels는 클래스 레이블, positive_class는 양성 클래스로 지정할 클래스를 나타냅니다. 이렇게 하면 해당 클래스에 대한 이진 분류자가 독립적으로 학습되며 해당 클래스와 다른 모든 클래스에 대한 판별 경계를 설정하게 됩니다.

>'OvR' 방식을 사용할 때, 모델은 각 클래스를 다른 모든 클래스와 비교하여 양성 클래스와 음성 클래스로 분류합니다. 분류기는 각 클래스에 대한 이진 분류를 수행하므로 클래스 순서에 따라 첫 번째 클래스가 양성 클래스로 선택됩니다. 따라서 코드에서 classes 리스트에 나열된 클래스 중 첫 번째 클래스, 즉 'Bream'이 양성 클래스로 선택됩니다. 이러한 동작은 모델이 어떤 클래스를 양성 클래스로 인식하는지에 영향을 미치며, 다른 클래스를 양성 클래스로 인식하려면 classes 리스트에서 다른 순서로 클래스를 나열하거나, 모델을 다시 훈련시키는 과정에서 클래스 순서를 변경할 수 있습니다.
