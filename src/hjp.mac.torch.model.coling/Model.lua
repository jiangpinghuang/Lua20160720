local Model = torch.class('PIT.Model')

function Model:__init(config)
  self.dim    = config.dim    or 150
end

function sentLSTM()
end

function charCNN()
end

function PIT.wordCNN()
  local inputSize = 300
  local outputSize = 200
  local kW = 1
  local dW = 1

  cnn1 = nn.Sequential()
  cnn1:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  cnn1:add(nn.ReLU())
  cnn1:add(nn.Max(1))

  cnn2 = nn.Sequential()
  cnn2:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  cnn2:add(nn.ReLU())
  cnn2:add(nn.Max(1))

  cnnp = nn.ParallelTable()
  cnnp:add(cnn1)
  cnnp:add(cnn2)

  cnn = nn.Sequential()
  cnn:add(cnnp)
  cnn:add(nn.CosineDistance())
  
  return cnn
end

function PIT.createModel()
end

function PIT.demoCNN(train, dev, embVec, model)  
  for i = 1, train.size do
    local lsent, rsent = train.lsent[i], train.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    local rinputs = embVec:index(1, rsent:long()):double()
    local input = {linputs, rinputs}
    local gold = torch.Tensor(1):fill(train.labels[i])
    local pred = model:forward(input)
    
    local criterion = nn.MSECriterion()
    local err = criterion:forward(pred, gold)
    local gradCriterion = criterion:backward(pred, gold)
    model:zeroGradParameters()
    model:backward(input, gradCriterion)
    model:updateParameters(0.01)
  end
  
  local devlabels = torch.Tensor(dev.size)
  local predictions = torch.Tensor(dev.size)
  for i = 1, dev.size do
    local lsent, rsent = dev.lsent[i], dev.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    local rinputs = embVec:index(1, rsent:long()):double()
    devlabels[i] = torch.Tensor(1):fill(dev.labels[i])
   
    local input = {linputs, rinputs}
    predictions[i] = model:forward(input)
  end

  local score = PIT.pearson(predictions, devlabels)
  return score
end

function PIT.predTest(model, test, embVec)
  local testlabels = torch.Tensor(test.size)
  local predictions = torch.Tensor(test.size)
  for i = 1, test.size do
    local lsent, rsent = test.lsent[i], test.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    local rinputs = embVec:index(1, rsent:long()):double()
    testlabels[i] = torch.Tensor(1):fill(test.labels[i])
   
    local input = {linputs, rinputs}
    predictions[i] = model:forward(input)
  end

  local score = PIT.pearson(predictions, testlabels)
  return score
end