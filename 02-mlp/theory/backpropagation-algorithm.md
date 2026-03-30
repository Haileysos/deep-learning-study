
#  Backpropagation Algorithm (역전파 학습 알고리즘)
: 경사 하강법을 사용하여 MLP의 가중치를 변경하는 학습 알고리즘  


무식한 경사 하강법 : 모든 가중치 위치에서 가중치를 앞뒤로 조금씩 변경시켜가며
손실함수를 계산하여 ∂E/∂w 를 계산한다. w1부터 시작하여 순서대로 모든 가중치에 적용하며,
가중치의 개수만큼 손실함수가 중복 계싼되어서 계산시간이 아주 많이 소모됨

역전파 알고리즘: 출력측에서 손실함수를 1번만 계산한 후에 이것을 신경망으로 역전파하여
모든 위치에서의 ∂E/∂w 를 계산한다. 이 과정에서 chain rule이 사용된다.
<img width="294" height="93" alt="image" src="https://github.com/user-attachments/assets/e0fd81b4-fbe3-42e0-9b2e-b9b2b01d54a6" />



출력층 유닛의 경우
<img width="592" height="373" alt="image" src="https://github.com/user-attachments/assets/93f7b30c-f5f4-4164-bcc3-ac2f1300ca2a" />

은닉층 유닛의 경우


Example 역전파 알고리즘을 손으로 계산해보자
<img width="493" height="242" alt="image" src="https://github.com/user-attachments/assets/fc446981-188d-46a9-9466-e54089b83ac2" />
순방향 패
