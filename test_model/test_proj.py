from test_model.model import GPT
import torch


model1 = GPT()
model2 = GPT()

p = 1

torch.save(model1.proj.state_dict(), './model1_weights.pth')
model2.proj.load_state_dict(torch.load('./model1_weights.pth'))
sum2 = count(model2)

p = 1