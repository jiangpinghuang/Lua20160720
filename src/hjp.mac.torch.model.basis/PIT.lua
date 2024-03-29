--[[
This model is for SemEval-2015 Task1 paraphrase identification and similarity measurement in Twitter. 
]]--

-- import packages --
require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

-- define a table --
PIT = {}

-- import a file into PIT --
include('Dict.lua')

-- define a header --
local function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end 

-- computing pearson correlation --
local function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm()) 
end

-- split sentence into tokens --
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

-- read embedding --
function PIT.readEmb(voc, emb)
  local vocab = PIT.Dict(voc)
  local embed = torch.load(emb)
  return vocab, embed
end

-- read sentences --
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

-- read train, dev and test data --
function PIT.readData(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsent = PIT.readSent(dir .. 'ls.txt', vocab)
  local linputs = s
  dataset.rsent = PIT.readSent(dir .. 'rs.txt', vocab)
  dataset.size  = #dataset.lsent
  local id = torch.DiskFile(dir .. 'id.txt')
  local sim = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)  
  for i = 1, dataset.size do
    dataset.ids[i] = id:readInt() 
    dataset.labels[i] = sim:readDouble()
  end  

  id:close()
  sim:close()  
  return dataset
end

-- PIT initialization --
function PIT:__init(config)
  self.layer            = config.layer            or 1
  self.dim              = config.dim              or 300
  self.learningRate     = config.learningRate     or 0.01
  self.epoch            = config.epoch            or 50
  self.batchSize        = config.batchSize        or 25
end

-- model configuration --
local function config()
  local layer = 1
  local dim = 300
  local learningRate = 0.01
  local epoch = 50
  local batchSize = 25
end

-- model train module, not valid, with cross-validation --
function PIT.train(train)

end

-- model train module with valid data --
function PIT.trainDev(train, dev, embVec)
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

  val = pearson(predictions, devlabels)
  print('val: ')
  print(val)
end

-- predict module with test data and trained model --
function PIT.predict(model, test)

end

-- save the trained parameter when obtained the best performance --
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

-- define a main module with various inputs --
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
  local trainDir = '/home/hjp/Workshop/Model/coling/pits/train/'
  local devDir = '/home/hjp/Workshop/Model/coling/pits/dev/'
  local testDir = '/home/hjp/Workshop/Model/coling/pits/test/'
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

  local epoches = 10
  for i = 1, epoches do
    timer = torch.Timer()
    PIT.trainDev(trainSet, testSet, vecs)
    print('Time elapsed for this epoch: ' .. timer:time().real .. ' seconds.')
  end 
  
  header('demo')
end

-- begin working --
main()