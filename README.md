# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/1baf6a15-d9a5-4ee8-88c3-e280cdbf3dd9" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Name: SUBHASH V

### Register Number: 212224240163

```python
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,8)
        self.n2=nn.Linear(8,10)
        self.n3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
sub=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sub.parameters(),lr=0.001)

def train_model(sub, X_train, y_train, criterion, optimizer, epochs=1000):
    # initialize history before loop
    sub.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = sub(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record loss
        sub.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="768" height="375" alt="image" src="https://github.com/user-attachments/assets/7e81c46b-e278-40d6-b0fa-a78286d4ccfb" />


## OUTPUT

<img width="432" height="242" alt="image" src="https://github.com/user-attachments/assets/d1a2b806-f492-4329-977c-450b3ca5bc60" />



### Training Loss Vs Iteration Plot

<img width="721" height="572" alt="image" src="https://github.com/user-attachments/assets/db954f05-29bc-4180-b921-d6a78e475542" />


### New Sample Data Prediction
```
X_new = torch.tensor([[9]], dtype=torch.float32)
X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

prediction = subhash(X_new_scaled).item()
print(f"Prediction for Input = 9 : {prediction}")
```
<img width="451" height="38" alt="image" src="https://github.com/user-attachments/assets/fd9d074e-247a-48d5-b547-38e91625cef7" />


## RESULT

Successfully executed the code to develop a neural network regression model.
