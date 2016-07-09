--[[
  Training script for semantic relatedness prediction on the SemEval2015 task1 PIT data set.
--]]

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')


similarityMeasure = {}      -- The important data structure table {} which wasn't understand when I learned Lua --

include('read_data.lua')    -- add all modules and function into similarityMeasure --
include('Vocab.lua')        -- add all modules and function into similarityMeasure --
include('Conv.lua')         -- add all modules and function into similarityMeasure --
include('CsDis.lua')        -- add all modules and function into similarityMeasure --

-- print similarityMeasure --
print('first')
print(similarityMeasure)

-- include('PaddingReshape.lua')
--printf = utils.printf       -- what's the utils?

-- global paths (modify if desired)
similarityMeasure.data_dir        = '/home/hjp/Workshop/Model/coling/'
similarityMeasure.models_dir      = '/home/hjp/Workshop/Model/coling/trained_models'
similarityMeasure.predictions_dir = '/home/hjp/Workshop/Model/coling/predictions'

-- print similarityMeasure --
print(similarityMeasure)

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
--local args = lapp [[
--Training script for semantic relatedness prediction on the PIT data set.
--  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
--  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
--  -d,--dim    (default 150)        LSTM memory dimension
--]]

local model_name, model_class, model_structure
model_name = 'convOnly'
model_class = similarityMeasure.Conv
model_structure = model_name

torch.seed()
-- torch.manualSeed(-3.0753778015266e+18)
print('<torch> using the automatic seed: ' .. torch.initialSeed())      -- why does the model use an initial seed? --

-- directory containing data set files
local data_dir = '/home/hjp/Workshop/Model/coling/pitt/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocabs.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = '/home/hjp/Workshop/Model/coling/data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')  -- both read_data.lua and Vocab.lua describe the function of read_embedding() --

local emb_dim = emb_vecs:size(2)  -- emb_vecs represents embedding matrix, size() means row * column, so size(2) is the dimension of vector --
--print('row:')
print(emb_vecs:size(1))
--print('column:')
print(emb_vecs:size(2))

-- use only vectors in vocabulary (not necessary, but gives faster training)
-- local oov_file = '/home/hjp/Downloads/oov.txt' --
-- ovf = io.open(oov_file, "a")
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do    -- load vocab-cased.txt, the file contains all words which distinguished lower and upper case letter -- 
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    -- print(w)
    -- print(emb_vocab:index(w))
    -- print(emb_vecs[emb_vocab:index(w)])
    vecs[i] = emb_vecs[emb_vocab:index(w)]    -- obtain the index of word w in emb_vacab, then read vector of this word via index --
  else
    --print(w)
    --ovf:write(w .."\n")
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)  -- a value between -0.05 and 0.05 is given for each elements in vecs[i] --
  end
end
--print('vecs: ')
--print(vecs)
print('vocab.size = ' .. vocab.size)
print('unk count = ' .. num_unk)
print('oov rate = ' .. num_unk / vocab.size)  -- oov_rate = 13.466621484243%
emb_vocab = nil     -- clear -- 
emb_vecs = nil      -- clear --
collectgarbage()
local taskD = 'sic'
-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD) -- read_data.lua implements the function --
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)     -- read_data.lua implements the function --
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)   -- read_data.lua implements the function --
--printf('num train = %d\n', train_dataset.size)
--printf('num dev   = %d\n', dev_dataset.size)
--printf('num test  = %d\n', test_dataset.size)

-- initialize model
print('model:')
print(model_class)
print('123')
local model = model_class{ -- Conv.lua contains the element which model needs, the following are initialed with input --
  emb_vecs   = vecs,
  structure  = model_structure, -- 'convOnly'
  num_layers = 1,--args.layers,
  mem_dim    = 150, --args.dim,
  task       = taskD,
}
print('abc')
print('model:')
print(model)

-- number of epochs to train
local num_epochs = 5

-- print information
header('model configuration')
--printf('max epochs = %d\n', num_epochs)
model:print_config()

if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
-- torch.setnumthreads(4)
-- print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model')

local id = 10005
print("Id: " .. id)
for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------') 
  model:trainCombineOnly(train_dataset)                     -- it cost much time at each epoch --
  print('Finished epoch in ' .. ( sys.clock() - start) )    -- better to print the time via dd:hh:mm:ss --
  
  local dev_predictions = model:predict_dataset(dev_dataset)
  print('dev_predictions: ')
  print(dev_predictions)
  print('dev_dataset.labels: ')
  print(dev_dataset.labels)
  local dev_score = pearson(dev_predictions, dev_dataset.labels)
  --printf('-- dev score: %.5f\n', dev_score)

  if dev_score >= best_dev_score then
    best_dev_score = dev_score
    local test_predictions = model:predict_dataset(test_dataset)
    local test_sco = pearson(test_predictions, test_dataset.labels)
    --printf('[[BEST DEV]]-- test score: %.4f\n', pearson(test_predictions, test_dataset.labels))

    local predictions_save_path = string.format(
    similarityMeasure.predictions_dir .. '/results-%s.%dl.%dd.epoch-%d.%.5f.%d.pred', 'ConvOnly', "1", "300", i, test_sco, id)
    local predictions_file = torch.DiskFile(predictions_save_path, 'w')
    print('writing predictions to ' .. predictions_save_path)
    for i = 1, test_predictions:size(1) do
      predictions_file:writeFloat(test_predictions[i])    -- write float data into test_predictions -- 
    end
    predictions_file:close()
  end
end
print('finished training in ' .. (sys.clock() - train_start))
