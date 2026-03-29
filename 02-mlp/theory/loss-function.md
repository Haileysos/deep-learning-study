## 4. Loss Function (손실함수) 계산  
: 신경망에서 학습을 시킬 때는 실제 출력과 원하는 출력 사이의 오차를 이용함.  
이 오차를 계산하는 함수를 손실함수 라고 함 
- 학습의 성과를 나타내는 지표이기도 함  
<img width="537" height="177" alt="image" src="https://github.com/user-attachments/assets/e703554d-5a39-40e2-9ed3-07168938f082" />  

### 가중치
- 가중치를 조절한다는 것은 스피커에서 나는 소리를 들으며 튜너 다이얼을 돌리는 것과 같음  
<img width="783" height="347" alt="image" src="https://github.com/user-attachments/assets/3ed1246e-a835-49ac-b6dc-6538bbc72abf" />

### MSE (평균 제곱 오차)
: 예측값과 정답값 간의 평균 제곱 오차  
<img width="293" height="86" alt="image" src="https://github.com/user-attachments/assets/5944637d-6144-4e8d-93e3-a197e8a12d19" />  
<img width="595" height="343" alt="image" src="https://github.com/user-attachments/assets/c95c9a8a-e421-45c6-afb7-3ceb872df7a2" />  
<img width="592" height="152" alt="image" src="https://github.com/user-attachments/assets/56b60501-0c8a-46c5-b540-b1922b3777f6" />  
- 예측값과 정답값이 차이가 많이 나는 경우  
<img width="593" height="147" alt="image" src="https://github.com/user-attachments/assets/e049821d-bf11-444a-a030-bb4b2ed4f61d" />  

<br><br><br>
---
<br><br><br>


## 5. Gradient Descent (경사 하강법)
: 손실 함수의 기울기(gradient)를 이용하여 손실값이 감소하는 방향으로 파라미터(가중치)를 반복적으로 업데이트하여 최솨값을 찾는 최적화 방법

@@@딥러닝 MLP_r1 PDF의 25/52 페이지 부터 ㄱㄱ

