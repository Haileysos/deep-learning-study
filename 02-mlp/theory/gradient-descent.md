# Gradient Descent (경사 하강법)  
: 손실함수의 기울기를 이용하여 손실값이 감소하는 방향으로 가중치(파라미터)를 반복적으로 업데이트하여 최소값을 찾는 최적화 방법

- 역전파 알고리즘은 신경망 학습 문제를 최적화 문제(optimization)로 접근함  
- 우리의 목표는 손실함수 값을 최소로 하는 가중치를 찾는 것  
<br><img width="297" height="183" alt="image" src="https://github.com/user-attachments/assets/4f90a9a5-f067-4206-9609-15287916ca38" /><br>  
    - 기울기가 양수 → 오른쪽으로 갈수록 올라감  ∴ 왼쪽으로 가야 내려감 → 가중치를 감소시켜야 함  
    - 기울기가 음수 → 오른쪽으로 갈수록 내려감  ∴ 오른쪽으로 가야 내려감 → 가중치를 증가시켜야 함  

###### 손실함수를 가중치로 미분한 값 = 손실함수를 x로 미분한 값 = 기울기 y  
<br>  

## Example  
경사하강법을 사용하여 위 함수의 최소값을 찾는 파이썬 프로그램을 작성하시오.  

<br><img width="777" height="348" alt="image" src="https://github.com/user-attachments/assets/d1d300a1-2f96-48f3-aad7-e56c8ed6ef15" /><br>  

수학적으로는 이 함수가 x = 3 일 때, 기울기가 0 이 되며, 최소값 y = 10 을 갖는다는 것을 알 수 있음
그러나 신경망에서는 이러한 값을 직접 구하지 않고 **기울기를 활용해** 손실함수 값을 최소로 하는 가중치를 찾아가야 함

```python
x = 10 # 처음 시작할 값 
learning_rate = 0.2 # 학습률
precision = 0.00001
max_iterations = 100 # 최대 반복 횟수

# 손실함수를 람다식으로 정의
loss_func = lambda x: (x-3)**2 + 10

# 그래디언트를 람다식으로 정의 (손실함수의 미분값)  
gradient = lambda x: 2 * x - 6

# 경사 강하법
for i in range(max_iterations):
  x = x - learning_rate * gradient(x) # x 값 업데이트
  print("손실함수값(", x, ") =", loss_func(x)) # 함수값 출력

print("최소값 = ", x)
```
