require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

PIT = {}

include('Model.lua')
include('Util.lua')
include('Vocab.lua')

config = {
  dim       = 100,
  learnRate = 0.01,
  batchSize = 5,
  layerSize = 1,
  regular   = 1e-4,
  modelName = 1,
  charCNN   = 1,
  wordCNN   = 1,
  sentLSTM  = 1,
  hidSize   = 150,
  epochSize = 10,
  crit      = nn.DistKLDivCriterion(),
  vDir      = '/home/hjp/Workshop/Model/coling/pit/vocabs.txt',
  eVocDir   = '/home/hjp/Workshop/Model/coling/vec/twitter.vocab',
  eDimDir   = '/home/hjp/Workshop/Model/coling/vec/twitter.th',
  trainDir  = '/home/hjp/Workshop/Model/coling/pit/train/',
  devDir    = '/home/hjp/Workshop/Model/coling/pit/dev/',
  testDir   = '/home/hjp/Workshop/Model/coling/pit/test/'  
}

local function train()
  local voc = PIT.vocab(config.vDir)
  local eVoc, eVec = PIT.readEmb(config.eVocDir, config.eDimDir)
  local vec = torch.Tensor(voc.size, eVec:size(2))
  for i = 1, voc.size do
    local t = voc:token(i)
    if eVoc:contains(t) then
      vec[i] = eVec[eVoc:index(t)]
    else
      vec[i]:uniform(-0.05, 0.05)
    end
  end
  eVoc, eVec = nil, nil
  collectgarbage()
  
  local trainSet  = PIT.readData(config.trainDir, voc)
  local devSet    = PIT.readData(config.devDir, voc)
  local bestScore = 0.0
  local bestModel = model
  for j = 1, config.epochSize do 
    timer = torch.Timer()
    local score = PIT.demoCNN(trainSet, devSet, vec)
    if score >= bestScore then
      bestScore = score
      bestModel = model
    end
    print(string.format("Epoch%3d, pearson: %6.8f, and costs %6.8f s.",j, score, timer:time().real))
  end
end

local function test()
  local testSet   = PIT.readData(config.testDir, voc) 
end

local function save()
  -- save the parameters and results.
end

local function main()
  train()
end

main()
