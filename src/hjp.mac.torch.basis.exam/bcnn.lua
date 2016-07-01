require('nn')
require('torch')

local inputSize = 300
local outputSize = 300
local kW = 1
local dW = 1

cnn1 = nn.Sequential()
cnn1:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
cnn1:add(nn.Tanh())
cnn1:add(nn.Max(1))

cnn2 = nn.Sequential()
cnn2:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
cnn2:add(nn.Tanh())
cnn2:add(nn.Max(1))

cnnp = nn.ParallelTable()
cnnp:add(cnn1)
cnnp:add(cnn2)

cnn = nn.Sequential()
cnn:add(cnnp)
cnn:add(nn.CosineDistance())

input = {torch.randn(4, 300), torch.randn(5, 300)}
output = cnn:forward(input)

for i = 1, 10 do
  x = {torch.randn(5, 300), torch.randn(8, 300)}
  y = torch.Tensor(1):fill(0.5)
  pred = cnn:forward(x)
  print('y:')
  print(y)
  print('pred: ')
  print(pred)
  criterion = nn.MSECriterion()
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  cnn:zeroGradParameters()
  cnn:backward(x, gradCriterion)
  cnn:updateParameters(0.05)
end

--for i = 1, 10 do 
--  i = {torch.randn(4, 300), torch.randn(6, 300)}
--  o = cnn:forward(i)
--  print('test: ')
--  print(o)
--end
