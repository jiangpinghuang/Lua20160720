-- SemEval-2015 Task1 PIT

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

PIT = {}

include('Dict.lua')

local function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end 

local function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm()) 
end

function PIT.sentSplit(sent, sep)
  local tokens = {}  
  while (true) do
    local pos = string.find(sent, sep)
    if (not pos) then
      tokens[#tokens + 1] = sent
      break
    end
    local token = string.sub(sent, 1, pos - 1)
    tokens[#tokens + 1] = token
    sent = string.sub(sent, pos + 1, #sent)
  end  
  return tokens
end

function PIT.readEmb(voc, emb)
  local vocab = PIT.Dict(voc)
  local embed = torch.load(emb)
  return vocab, embed
end

function PIT.readSent(path, vocab)
  local sents = {}
  local file = io.open(path, 'r')
  local line  
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = PIT.sentSplit(line, " ")
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sents[#sents + 1] = sent
  end  
  file:close()
  return sents
end

function PIT.readData(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsent = PIT.readSent(dir .. 'a.txt', vocab)
  local linputs = s
  print(dataset.lsent)
  dataset.rsent = PIT.readSent(dir .. 'b.txt', vocab)
  print(dataset.rsent)
  dataset.size  = #dataset.lsent
  local id = torch.DiskFile(dir .. 'id.txt')
  local sim = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)  
  for i = 1, dataset.size do
    dataset.ids[i] = id:readInt() 
    dataset.labels[i] = sim:readDouble()
  end  
  print('lsent: ')
  print(dataset.lsent[1])
  print('rsent: ')
  print(dataset.rsent[1])
  print('ids: ')
  print(dataset.ids)
  print('sim: ')
  print(dataset.labels)
  id:close()
  sim:close()  
  return dataset
end

function PIT:__init(config)
  self.layer            = config.layer            or 1
  self.dim              = config.dim              or 300
  self.learningRate     = config.learningRate     or 0.01
  self.epoch            = config.epoch            or 50
  self.batchSize        = config.batchSize        or 25
end

local function config()
  local layer = 1
  local dim = 300
  local learningRate = 0.01
  local epoch = 50
  local batchSize = 25
end

function PIT.train(train)

end

function PIT.trainDev(train, dev, embVec)
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
  


  --input = {torch.randn(4, 300), torch.randn(5, 300)}
  ---output = cnn:forward(input)

  for i = 1, train.size do
    local lsent, rsent = train.lsent[i], train.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    --print('linputs: ')
    --print(linputs)  
    --print('rinputs: ')
    local rinputs = embVec:index(1, rsent:long()):double()
   -- print(rinputs)
   -- print('sim: ')
   -- print(i)
   -- print(train.labels[i])
    --local x = {linputs, rinputs}
    input = {linputs, rinputs}
    --input = {torch.randn(7, 300), torch.randn(12, 300)}
    y = torch.Tensor(1):fill(train.labels[i])
    print('y:')
    print(y)
    --print(linputs)
    pred = cnn:forward(input)
    print('pred: ')
    print(pred)
    print('err: ')
    
    criterion = nn.MSECriterion()
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    cnn:zeroGradParameters()
    cnn:backward(input, gradCriterion)
    cnn:updateParameters(0.05)
    print(err)
    --local err = criterion:forward(pred, y)

    
    --criterion = nn.MSECriterion()
    --local err = criterion:forward(pred, y)
   -- local gradCriterion = criterion:backward(pred, y)
    --cnn:zeroGradParameters()
   -- cnn:backward(x, gradCriterion)
   -- cnn:updateParameters(0.05)
  end
  
  local devlabels = torch.Tensor(dev.size)
  local predictions = torch.Tensor(dev.size)
  for i = 1, dev.size do
    local lsent, rsent = dev.lsent[i], train.rsent[i]
    local linputs = embVec:index(1, lsent:long()):double()
    --print('linputs: ')
    --print(linputs)  
    --print('rinputs: ')
    local rinputs = embVec:index(1, rsent:long()):double()
    --print(rinputs)
    print('sim: ')
    devlabels[i] = torch.Tensor(1):fill(dev.labels[i])
   
    local x = {linputs, rinputs}
    --local y = dev.labels[i] * 0.2
    pred = cnn:forward(x)
    predictions[i] = cnn:forward(x)
    print("dev: ")
    print(pred)
  end
  print('devlabels: ')
  print(devlabels)
  print('predictions: ')
  print(predictions)
  print('pearson: ')
  val = pearson(predictions, devlabels)
  print('val: ')
  print(val)
end

function PIT.predict(model, test)

end

function PIT.save(model)
  local config = {
    layer         = self.layer,
    dim           = self.dim,
    learningRate  = self.learningRate,
    epoch         = self.epoch,
    batchSize     = self.batchSize,
  }
  torch.save(model, {
    params = self.params,
    config = config,
  })
end

local function main()
  header('Loading vectors...')
  local vocDir = '/home/hjp/Workshop/Model/coling/pit/vocabs.txt'
  local vocab = PIT.Dict(vocDir)
  local eVocDir = '/home/hjp/Workshop/Model/coling/vec/twitter.vocab'
  local eDimDir = '/home/hjp/Workshop/Model/coling/vec/twitter.th'
  local eVoc, eVec = PIT.readEmb(eVocDir, eDimDir)
  local dimSize = eVec:size(2)
  
  local vecs = torch.Tensor(vocab.size, dimSize)
  for i = 1, vocab.size do
    local w = vocab:token(i)
    if eVoc:contains(w) then
      vecs[i] = eVec[eVoc:index(w)]
    else
      vecs[i]:uniform(-0.05, 0.05)
    end
  end

  eVoc, eVec = nil, nil
  collectgarbage()

  header('Loading datasets...')
  local trainDir = '/home/hjp/Workshop/Model/coling/pit/train/'
  local devDir = '/home/hjp/Workshop/Model/coling/pit/dev/'
  local testDir = '/home/hjp/Workshop/Model/coling/pit/test/'
  local trainSet = PIT.readData(trainDir,vocab)
  local devSet = PIT.readData(devDir,vocab)
  local testSet = PIT.readData(testDir,vocab)
  print('train size: ' .. trainSet.size)
  print('dev size: ' .. devSet.size)
  print('test size: ' .. testSet.size)
  
  local modelName, modelClass, modelStruct
  modelName   = 'CNN'
  modelClass  = PIT.ConvNN
  modelStruct = modelName
  
  local config = {
    layer = 1,
    dims  = 300
  }
  
--  model = model_class{
--    embVec  = vecs,
--    struct  = modelStruct,
--    layers  = config.layer,
--    vecDim  = config.dims
--  }
  
  PIT.trainDev(trainSet,devSet, vecs)
  
  
  header('demo')
end

main()