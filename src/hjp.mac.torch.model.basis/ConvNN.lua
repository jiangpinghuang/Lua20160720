local ConvNN = torch.class('PIT.ConvNN')

function ConvNN:__init(config)
  self.dim    = config.dim    or 150
  self.lr     = config.lr     or 0.01
  self.bs     = config.bs     or 5
  self.layer  = config.layer  or 1
  self.reg    = config.reg    or 1e-4
  self.struct = config.struct or 'LSTM'
  self.hid    = config.hid    or 150
  
  self.vec    = config.vec
  self.dim    = config.vec:size(2)
  
  self.optim  = {lr = self.lr}
  self.crit   = nn.DistKLDivCriterion()
end

function pitLookUp(embVec)
  if embVec == nil then
    print('embVec null!')
  end
  
  local vocSize = embVec:size(1)
  local dim = embVec:size(2)
  local txt = nn.Squential()
  txt:add(nn.LookupTable(vocSize, dim))
  
  for i = 1, vocSize do
    if 1 == 2 then
      txt:get(1).weight[i] = torch.randn(dim)
    else
      local emb = torch.Tensor(dim):copy(embVec[i])
      txt:get(1).weight[i] = emb
    end
  end
  
  local model = nn.Sequential()
  if 1 == 2 then
    model = txt:clone()
  else
    model = txt:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
  local deep = nn.Sequential()
  query = nn.ParallelTable()
  query:add(model)
  query:add(txt)
  return deep
end
   
function pitModel(model, vocSize, dimSize, nOut, kkW)
  local net = nn.Sequential()
  local txt = nn.Sequential()
  local cat = nn.Sequential()
  
  local con1 = nn.Sequential()
  local con2 = nn.Sequential()
  local con3 = nn.Sequential()
  local con4 = nn.Sequential()
  
  local cat1 = nn.Concat(1)
  local cat2 = nn.Concat(1)
  local cat3 = nn.Concat(1)
  local cat4 = nn.Concat(1)
  local cat5 = nn.Concat(1)
  
end
  