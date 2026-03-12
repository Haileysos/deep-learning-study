
### AND 게이트

```python
import numpy as np

epsilon = 0.0000001

def perceptron (x1, x2):

    X = np.array([x1,x2])
    W = np.array([1.0,1.0])
    B = -1.5

    sum = np.dot(W,X) + B

    if sum > epsilon:
        return 1
    else:
        return 0

input1, input2 = [i for i in map(int, input().split())]
print(perceptron(input1, input2)
```  
---

### Perceptron learning algorithm

```python


```
