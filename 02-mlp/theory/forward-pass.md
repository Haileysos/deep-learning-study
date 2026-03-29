
#  MLP의 순방향 패스  
: 입력 신호가 입력층 유닛에 가해지고 이 입력 신호가 은닉층을 통해 출력층으로 전파되는 과정
<img width="583" height="437" alt="image" src="https://github.com/user-attachments/assets/2bae0fd4-a627-4c0a-9a86-bdb2e6758589" />  
손계산 Time~  
<img width="529" height="288" alt="image" src="https://github.com/user-attachments/assets/eeedae6b-2be6-4738-9ad8-fe9be6e9a4b5" />
<img width="327" height="162" alt="image" src="https://github.com/user-attachments/assets/9cf5d9c8-eb7a-4736-909f-e5340397dd25" />
<img width="677" height="221" alt="image" src="https://github.com/user-attachments/assets/372da7df-0d70-4033-9071-90cfec03705f" />
<img width="433" height="213" alt="image" src="https://github.com/user-attachments/assets/002b3a0b-6137-4e5a-a78a-b849472238fc" />  
정답은 0이지만 신경망의 출력은 0.71 정도이다. 오차가 상당함을 알 수 있다.  

행렬로 표시해보자 Time~    
<img width="686" height="367" alt="image" src="https://github.com/user-attachments/assets/97bb0b2a-ed6a-47d6-af0b-edb938dc0a80" />  

MLP 순방향 패스를 코딩해보자 실습~
```python
import numpy as np

# 시그모이드 함수
def actf(x):
	return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분치
def actf_deriv(x):
  return x * (1 - x)

# 입력 유닛의 개수, 은닉 유닛의 개수, 출력 유닛의 개수
inputs, hiddens, outputs = 2, 2, 1
learning_rate = 0.2

# 훈련 샘플과 정답
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

W1 = np.array([[0.10, 0.20],
               [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

# 순방향 전파 계산
def predict(x):
  layer0 = x                   # 입력을 layer0에 대입
  Z1 = np.dot(layer0, W1) + B1 # 행렬의 곱 계산
  layer1 = actf(Z1)            # 활성화 함수 적용
  Z2 = np.dot(layer1, W2) + B2 # 행렬의 곱 계산
  layer2 = actf(Z2)            # 활성화 함수 적용
  return layer0, layer1, layer2

def test():
  for x, y in zip(X, T):
    x = np.reshape(x, (1, -1)) # x 를 2차원 행렬로 만든다. 입력은 2차원이어야 한다.
    layer0, layer1, layer2 = predict(x)
    print(x, y, layer2)
test()
```

<br><br><br>
---
<br><br><br>
