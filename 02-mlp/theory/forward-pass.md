
#  MLP의 순방향 패스  
: 입력 신호가 입력층 유닛에 가해지고 이 입력 신호가 은닉층을 통해 출력층으로 전파되는 과정  
<br><img width="40%" alt="image" src="https://github.com/user-attachments/assets/2bae0fd4-a627-4c0a-9a86-bdb2e6758589" /><br>  
<br><br>  

## Example  

### ➡️ Manual Calculation  
<p>
  <img src="https://github.com/user-attachments/assets/eeedae6b-2be6-4738-9ad8-fe9be6e9a4b5" width="48%">
  <img src="https://github.com/user-attachments/assets/9cf5d9c8-eb7a-4736-909f-e5340397dd25" width="28%">
</p>
<br><img width="50%" alt="image" src="https://github.com/user-attachments/assets/372da7df-0d70-4033-9071-90cfec03705f" /><br>
<br><img width="35%" alt="image" src="https://github.com/user-attachments/assets/002b3a0b-6137-4e5a-a78a-b849472238fc" /><br>  
<br>  

정답은 `0` but 신경망의 실제 출력은 `0.71` 정도 
###### 오차가 상당함을 알 수 있음  
<br>   

### ➡️ Matrix Representation 
<br><img width="60%" alt="image" src="https://github.com/user-attachments/assets/97bb0b2a-ed6a-47d6-af0b-edb938dc0a80" /><br>  
<br>   

### ➡️ Coding the MLP Forward Pass
```python
import numpy as np

# 시그모이드 함수
def actf(x):
	return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분치
def actf_deriv(x):    # 역전파 할 떄 필요
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
  for x, y in zip(X, T): # X와 T를 짝지어서 하나씩 꺼냄 (입력 x 와 그에 해당하는 정답 y 를 같이 꺼냄)
    x = np.reshape(x, (1, -1)) # 원래 x = [0, 1] 은 방향이 없는 일렬 데이터로서 (2,) 형태라 애매함
                               # x 를 (2,1) 꼴의 2차원 행렬로 만들어야 (2,2)인 W1, (2,1)인 W2와 곱셈 가능
    layer0, layer1, layer2 = predict(x)
    print(x, y, layer2)
test()
```

<br><br><br>
---
<br><br><br>
