require 'nn'

str = "----------------------------------------"

print(str .. "CNN" .. str)
cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(1, 6, 5, 5))
cnn:add(nn.Tanh())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.SpatialConvolution(6, 16, 5, 5))
cnn:add(nn.Tanh())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.View(16*5*5))
cnn:add(nn.Linear(16*5*5, 120))
cnn:add(nn.Tanh())
cnn:add(nn.Linear(120, 84))
cnn:add(nn.Tanh())
cnn:add(nn.Linear(84, 10))
cnn:add(nn.LogSoftMax())

print('Lenet5\n' .. cnn:__tostring())

input = torch.rand(1, 32, 32)
output = cnn:forward(input)

print(input)
print(output)

cnn:zeroGradParameters()
gradInput = cnn:backward(input, torch.rand(10))
print(#gradInput)

criterion = nn.ClassNLLCriterion()
criterion:forward(output, 3)
gradients = criterion:backward(output, 3)
gradInput = cnn:backward(input, gradients)

m = nn.SpatialConvolution(1, 3, 2, 2)
print(m.weight)
print(m.bias)

print(str .. "End" .. str)