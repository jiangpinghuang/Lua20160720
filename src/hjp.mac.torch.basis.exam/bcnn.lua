require('nn')
require('torch')
require('cunn')

local inputSize = 10
local outputSize = 10
local kW = 1
local dW = 1

cnn = nn.Sequential()
cnn:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
cnn:add(nn.Tanh())
cnn:add(nn.Max(1))
--print(cnn)

input = torch.randn(4, 10)
--print(input)
output = cnn:forward(input)
--print(output)

for i = 1, 2000000 do
  x = torch.randn(4, 10)
  y = torch.ones(1, 10)
  --print(x)
  --print(y)
  pred = cnn:forward(x)
  --print(pred)
  criterion = nn.MSECriterion()
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  --print(err)
  --print(gradCriterion)
  cnn:zeroGradParameters()
  cnn:backward(x, gradCriterion)
  cnn:updateParameters(0.01)
  --print(err)
end

test = torch.randn(4, 10)

print(cnn:forward(test))