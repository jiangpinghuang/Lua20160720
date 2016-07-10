local model = torch.class('pit.model')

function model:__init(config)
  self.dim    = config.dim    or 150
end

function sentLSTM()
end

function charCNN()
end

function wordCNN()
end

function pit.demoCNN(train, dev, embVec)
  local inputSize = 300
  local outputSize = 300
  local kW = 1
  local dW = 1

  cnn1 = nn.Sequential()
  cnn1:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  cnn1:add(nn.PReLU())
  cnn1:add(nn.Max(1))

  cnn2 = nn.Sequential()
  cnn2:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  cnn2:add(nn.PReLU())
  cnn2:add(nn.Max(1))

  cnnp = nn.ParallelTable()
  cnnp:add(cnn1)
  cnnp:add(cnn2)

  cnn = nn.Sequential()
  cnn:add(cnnp)
  cnn:add(nn.CosineDistance())
  
  for i = 1, train.size do
    local lsent, rsent = train.lsent[i], train.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    local rinputs = embVec:index(1, rsent:long()):double()
    input = {linputs, rinputs}
    y = torch.Tensor(1):fill(train.labels[i])
    pred = cnn:forward(input)
    
    criterion = nn.MSECriterion()
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    cnn:zeroGradParameters()
    cnn:backward(input, gradCriterion)
    cnn:updateParameters(0.01)
  end
  
  local devlabels = torch.Tensor(dev.size)
  local predictions = torch.Tensor(dev.size)
  for i = 1, dev.size do
    local lsent, rsent = dev.lsent[i], train.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    local rinputs = embVec:index(1, rsent:long()):double()
    devlabels[i] = torch.Tensor(1):fill(dev.labels[i])
   
    local x = {linputs, rinputs}
    pred = cnn:forward(x)
    predictions[i] = cnn:forward(x)
  end

  val = pit.pearson(predictions, devlabels)
  return val
end