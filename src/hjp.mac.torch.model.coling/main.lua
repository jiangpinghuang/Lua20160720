require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

pit = {}

include('model.lua')
include('util.lua')
include('vocab.lua')

local function config()
end

local function train()
end

local function valid()
end

local function test()
end

local function save()
end

local function main()
  print("Loading vectors...")
  local vDir = '/home/hjp/Workshop/Model/coling/pit/vocabs.txt'
  local voc = pit.vocab(vDir)
  local eVocDir = "/home/hjp/Workshop/Model/coling/vec/twitter.vocab"
  local eDimDir = "/home/hjp/Workshop/Model/coling/vec/twitter.th"
  local eVoc, eVec = pit.readEmb(eVocDir, eDimDir)
  local dimSize = eVec:size(2)  
  local vec = torch.Tensor(voc.size, dimSize)
  for i = 1, voc.size do
    local w = voc:token(i)
    if eVoc:contains(w) then
      vec[i] = eVec[eVoc:index(w)]
    else
      vec[i]:uniform(-0.05, 0.05)
    end
  end  
  eVoc, eVec = nil, nil
  collectgarbage()
  print("Loaded " .. voc.size .. " words embedding!")
  
  print("Loading data...")
  local trainDir = '/home/hjp/Workshop/Model/coling/pits/train/'
  local devDir = '/home/hjp/Workshop/Model/coling/pits/dev/'
  local testDir = '/home/hjp/Workshop/Model/coling/pits/test/'
  local trainSet = pit.readData(trainDir,voc)
  local devSet = pit.readData(devDir,voc)
  local testSet = pit.readData(testDir,voc)
  print('train size: ' .. trainSet.size)
  print('dev size: ' .. devSet.size)
  print('test size: ' .. testSet.size)
  print("finished!")
  
  pit.CNN(trainSet, testSet, vec)
end

main()










































































