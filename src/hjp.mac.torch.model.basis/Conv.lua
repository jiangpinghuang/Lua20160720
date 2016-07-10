local Conv = torch.class('similarityMeasure.Conv')

-- initialization configuration for the model -- 
function Conv:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.01
  self.batch_size    = config.batch_size    or 5     --1 --25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 150
  self.task          = config.task          or 'sic'  -- or 'vid'
  
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  if self.task == 'sic' then
    self.num_classes = 5
  elseif self.task == 'vid' then
    self.num_classes = 6
  elseif self.task == 'pit' then      -- add pit task --
    self.num_classes = 1
  else
    error("not possible task!")
  end
  
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.DistKLDivCriterion()
  
  ----------------------------------------Combination of ConvNets----------------------------------------
  
  
  -- flexLookUpPoin for msrp task
function flexLookUpPoin(emb_vecs, poin_vecs)
  print('flexLookUpPoin..........................................')
  if emb_vecs == nil or poin_vecs == nil then
  error("Not good!")
  end
  local poinDim = poin_vecs:size(2)
  local vocaSize = emb_vecs:size(1)
  local Dim = emb_vecs:size(2)
  local featext = nn.Sequential()
  local lookupPara=nn.ParallelTable()             -- nn's modules must be studied recently --
  lookupPara:add(nn.LookupTable(vocaSize,Dim)) --numberWords*D
  lookupPara:add(nn.LookupTable(53,poinDim)) --numberWords*DPos
  featext:add(lookupPara)
  featext:add(nn.JoinTable(2)) -- numberWords*(D+DPos)  
  -----------------Initialization----------------
  for i=1, vocaSize do
  if 1 == 2 then
    featext:get(1):get(1).weight[i] = torch.randn(Dim)      
  else 
    local emb = torch.Tensor(Dim):copy(emb_vecs[i])
    featext:get(1):get(1).weight[i] = emb     
  end     
  end  
  
  for i=1, poin_vecs:size(1) do
  ----POIN parts!!  
  local emb2 = torch.Tensor(poinDim):copy(poin_vecs[i])
  featext:get(1):get(2).weight[i] = emb2
  end       
  ---------------------------CLONE!-------------
  local modelQ = nn.Sequential()
  if 1 == 2 then  
    modelQ= featext:clone()
  else
    modelQ= featext:clone('weight','bias','gradWeight','gradBias')
  end
  local deepM = nn.Sequential()
  paraQuery=nn.ParallelTable()
  paraQuery:add(modelQ)
  paraQuery:add(featext)            
  deepM:add(paraQuery)
  return deepM
end

print('flexLookUpPoin')

function flexLookUp(emb_vecs)
  if emb_vecs == nil then
  error("Not good!")
  end
  
  local vocaSize = emb_vecs:size(1)
  print('VocaSize: ' .. vocaSize)
  local Dim = emb_vecs:size(2)
  local featext = nn.Sequential()   
  featext:add(nn.LookupTable(vocaSize,Dim))
  -----------------Initialization----------------
  for i=1, vocaSize do
  if 1 == 2 then
    featext:get(1).weight[i] = torch.randn(Dim)     
  else 
    local emb = torch.Tensor(Dim):copy(emb_vecs[i])
    featext:get(1).weight[i] = emb
    ----No POIN parts!!     
  end     
  end  
  ---------------------------CLONE!-------------
  local modelQ = nn.Sequential()
  if 1 == 2 then  
    modelQ= featext:clone()
  else
    modelQ= featext:clone('weight','bias','gradWeight','gradBias')
  end
  local deepM = nn.Sequential()
  paraQuery=nn.ParallelTable()
  paraQuery:add(modelQ)
  paraQuery:add(featext)       
  deepM:add(paraQuery)  
  return deepM
end

print('flexLookUp')

function createModel(mdl, vocsize, Dsize, nout, KKw)
    -- define model to train
    local network = nn.Sequential()
    local featext = nn.Sequential()
    local classifier = nn.Sequential()

    local conCon1 = nn.Sequential()
    local conCon2 = nn.Sequential()
    local conCon3 = nn.Sequential()
    local conCon4 = nn.Sequential()

    local parallelConcat1 = nn.Concat(1)
    local parallelConcat2 = nn.Concat(1)
    local parallelConcat3 = nn.Concat(1)
    local parallelConcat4 = nn.Concat(1)
    local parallelConcat5 = nn.Concat(1)

    local D     = Dsize -- opt.dimension
    local kW    = KKw -- opt.kwidth
    local dW    = 1 -- opt.dwidth
    local noExtra = false
    local nhid1 = 250 -- opt.nhid1 
    local nhid2 = 250 -- opt.nhid2
    local NumFilter = D
    local pR = 2 -- opt.pR     -- if pR = 1, add(nn.PReLU()) --
    local layers=1
      
    if mdl == 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt' then
      
      local PaddingReshape, parent = torch.class('nn.PaddingReshape', 'nn.Module')

function PaddingReshape:__init(...)
   parent.__init(self)
   local arg = {...}

   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   if torch.type(arg[#arg]) == 'boolean' then
      self.batchMode = arg[#arg]
      table.remove(arg, #arg)
   end
   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n) 
      for i=1,n do --modifed index
         self.size[i] = arg[i] --modified shift
      end
   end

   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
   end
   
   -- only used for non-contiguous input or gradOutput
   self._input = torch.Tensor()
   self._gradOutput = torch.Tensor()
end

function PaddingReshape:updateOutput(input)
   if not input:isContiguous() then
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end
   
   argsYoshi = torch.LongStorage()
   local nsi = #input:size() --modified
   argsYoshi:resize(nsi+1) 
   argsYoshi[1] = 1
   for i=2,nsi+1 do --modifed index
    argsYoshi[i] = input:size()[i-1] --modified shift
   end
   self.batchMode = false
      
   if (self.batchMode == false) or (
         (self.batchMode == nil) and 
         (input:nElement() == self.nelement and input:size(1) ~= 1)
      ) then
      self.output:view(input, argsYoshi) --modified
   else
      self.batchsize[1] = input:size(1)
      self.output:view(input, self.batchsize)
   end
   return self.output
end

function PaddingReshape:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end


function PaddingReshape:__tostring__()
  return torch.type(self) .. '(Pad ' ..
      table.concat(self.size:totable(), 'x') .. ')'
end
      
      
    --dofile "PaddingReshape.lua"
    
    deepQuery=nn.Sequential()
    D = Dsize     -- 300 --
    print('Dsize: ')
    print(Dsize)

    print('NumFilter: ')
    print(NumFilter)

    print('dw: ')
    print(dw)
    
    print("kW: ")
    print(kW)
      
    -- max part for comparing --  
    local incep1max = nn.Sequential()
    incep1max:add(nn.TemporalConvolution(D,NumFilter,1,dw))
      if pR == 1 then
        incep1max:add(nn.PReLU())
      else 
        incep1max:add(nn.Tanh())
      end     
      incep1max:add(nn.Max(1))      -- the max value in column
      incep1max:add(nn.Reshape(NumFilter,1))      
      local incep2max = nn.Sequential()
      incep2max:add(nn.Max(1))
      incep2max:add(nn.Reshape(NumFilter,1))        
      local combineDepth = nn.Concat(2)   -- expand column by concating rows -- 
      combineDepth:add(incep1max)
      combineDepth:add(incep2max)
      
      local ngram = kW                
      for cc = 2, ngram do
        local incepMax = nn.Sequential()
        if not noExtra then
          incepMax:add(nn.TemporalConvolution(D,D,1,dw)) -- set
          if pR == 1 then
            incepMax:add(nn.PReLU())
          else 
            incepMax:add(nn.Tanh())
          end
        end
        incepMax:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
        if pR == 1 then
          incepMax:add(nn.PReLU())
        else 
          incepMax:add(nn.Tanh())
        end
        incepMax:add(nn.Max(1))
        incepMax:add(nn.Reshape(NumFilter,1))               
        combineDepth:add(incepMax)        
      end       
           
      -- min part for comparing --  
      local incep1min = nn.Sequential()
      incep1min:add(nn.TemporalConvolution(D,NumFilter,1,dw))
      if pR == 1 then
      incep1min:add(nn.PReLU())
      else 
      incep1min:add(nn.Tanh())
      end     
      incep1min:add(nn.Min(1))
      incep1min:add(nn.Reshape(NumFilter,1))      
      local incep2min = nn.Sequential()
      incep2min:add(nn.Min(1))
      incep2min:add(nn.Reshape(NumFilter,1))      
      combineDepth:add(incep1min)
      combineDepth:add(incep2min)
      
      for cc = 2, ngram do        
        local incepMin = nn.Sequential()
        if not noExtra then
        incepMin:add(nn.TemporalConvolution(D,D,1,dw)) --set
        if pR == 1 then
          incepMin:add(nn.PReLU())
        else 
          incepMin:add(nn.Tanh())
        end
        end     
        incepMin:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
        if pR == 1 then
          incepMin:add(nn.PReLU())
      else 
          incepMin:add(nn.Tanh())
      end
        incepMin:add(nn.Min(1))
        incepMin:add(nn.Reshape(NumFilter,1))               
        combineDepth:add(incepMin)                  
      end  
      
      -- mean part for comparing -- 
      local incep1mean = nn.Sequential()
      incep1mean:add(nn.TemporalConvolution(D,NumFilter,1,dw))
      if pR == 1 then
      incep1mean:add(nn.PReLU())
      else 
      incep1mean:add(nn.Tanh())
      end
      incep1mean:add(nn.Mean(1))
      incep1mean:add(nn.Reshape(NumFilter,1))                   
      local incep2mean = nn.Sequential()
      incep2mean:add(nn.Mean(1))
      incep2mean:add(nn.Reshape(NumFilter,1))     
      combineDepth:add(incep1mean)
      combineDepth:add(incep2mean)      
      for cc = 2, ngram do
        local incepMean = nn.Sequential()
        if not noExtra then
          incepMean:add(nn.TemporalConvolution(D,D,1,dw)) --set
          if pR == 1 then
          incepMean:add(nn.PReLU())
        else 
          incepMean:add(nn.Tanh())
        end
        end
        incepMean:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
        if pR == 1 then
        incepMean:add(nn.PReLU())
      else 
        incepMean:add(nn.Tanh())
      end
        incepMean:add(nn.Mean(1))
        incepMean:add(nn.Reshape(NumFilter,1))          
        combineDepth:add(incepMean) 
      end  
      
      local conceptFNum = 20
      for cc = 1, ngram do
        local perConcept = nn.Sequential()
        perConcept:add(nn.PaddingReshape(2,2)) --set
        perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,ngram)) --set
        perConcept:add(nn.Max(2)) --set
        if pR == 1 then
        perConcept:add(nn.PReLU())
      else 
        perConcept:add(nn.Tanh())
      end
      perConcept:add(nn.Transpose({1,2}))
        combineDepth:add(perConcept)  
      end
      for cc = 1, ngram do
        local perConcept = nn.Sequential()
        perConcept:add(nn.PaddingReshape(2,2)) --set
        perConcept:add(nn.SpatialConvolutionMM(1,conceptFNum,1,ngram)) --set
        perConcept:add(nn.Min(2)) --set
        if pR == 1 then
        perConcept:add(nn.PReLU())
      else 
        perConcept:add(nn.Tanh())
      end
      perConcept:add(nn.Transpose({1,2}))
        combineDepth:add(perConcept)  
      end
      
      featext:add(combineDepth)   
      local items = (ngram+1)*3     
      local separator = items+2*conceptFNum*ngram
      local sepModel = 0 
      if sepModel == 1 then  
      modelQ= featext:clone()
      else
      modelQ= featext:clone('weight','bias','gradWeight','gradBias')
      end
      paraQuery=nn.ParallelTable()
      paraQuery:add(modelQ)
          paraQuery:add(featext)      
          deepQuery:add(paraQuery) 
      deepQuery:add(nn.JoinTable(2)) 
      
      d=nn.Concat(1) 
      for i=1,items do
        if i <= items/3 then          
          for j=1,items/3 do
            --if j == i then
            local connection = nn.Sequential()
            local minus=nn.Concat(2)
            local c1=nn.Sequential()
            local c2=nn.Sequential()
            c1:add(nn.Select(2,i)) -- == D, not D*1
            c1:add(nn.Reshape(NumFilter,1)) --D*1 here          
            c2:add(nn.Select(2,separator+j))          
            c2:add(nn.Reshape(NumFilter,1))
            minus:add(c1)
            minus:add(c2)
            connection:add(minus) -- D*2            
            local similarityC=nn.Concat(1) -- multi similarity criteria     
            local s1=nn.Sequential()
            s1:add(nn.SplitTable(2))
            s1:add(nn.PairwiseDistance(2)) -- scalar
            local s2=nn.Sequential()
            if 1 < 3 then
              s2:add(nn.SplitTable(2))
            else
              s2:add(nn.Transpose({1,2})) 
              s2:add(nn.SoftMax())
              s2:add(nn.SplitTable(1))                    
            end           
            s2:add(nn.CsDis()) -- scalar
            local s3=nn.Sequential()
            s3:add(nn.SplitTable(2))
            s3:add(nn.CSubTable()) -- linear
            s3:add(nn.Abs()) -- linear            
            similarityC:add(s1)
            similarityC:add(s2)         
            similarityC:add(s3)
            connection:add(similarityC) -- scalar                     
            d:add(connection)
            -- end
          end
        elseif i <= 2*items/3 then        
          for j=1+items/3, 2*items/3 do
            -- if j == i then
            local connection = nn.Sequential()
            local minus=nn.Concat(2)
            local c1=nn.Sequential()
            local c2=nn.Sequential()
            c1:add(nn.Select(2,i)) -- == NumFilter, not NumFilter*1
            c1:add(nn.Reshape(NumFilter,1)) -- NumFilter*1 here
            c2:add(nn.Select(2,separator+j))
            c2:add(nn.Reshape(NumFilter,1))
            minus:add(c1)
            minus:add(c2)
            connection:add(minus) -- D*2            
            local similarityC=nn.Concat(1) -- multi similarity criteria     
            local s1=nn.Sequential()
            s1:add(nn.SplitTable(2))
            s1:add(nn.PairwiseDistance(2)) -- scalar
            local s2=nn.Sequential()      
            if 1 < 3 then
              s2:add(nn.SplitTable(2))
            else
              s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
              s2:add(nn.SoftMax())
              s2:add(nn.SplitTable(1))                    
            end                 
            s2:add(nn.CsDis()) -- scalar            
            local s3=nn.Sequential()
            s3:add(nn.SplitTable(2))
            s3:add(nn.CSubTable()) -- linear
            s3:add(nn.Abs()) -- linear            
            similarityC:add(s1)
            similarityC:add(s2)         
            similarityC:add(s3)
            connection:add(similarityC) -- scalar                       
            d:add(connection)
            -- end
          end
        else 
          for j=1+2*items/3, items do
            --if j == i then
            local connection = nn.Sequential()
            local minus=nn.Concat(2)
            local c1=nn.Sequential()
            local c2=nn.Sequential()
            c1:add(nn.Select(2,i)) -- == D, not D*1
            c1:add(nn.Reshape(NumFilter,1)) --D*1 here
            c2:add(nn.Select(2,separator+j))
            c2:add(nn.Reshape(NumFilter,1))
            minus:add(c1)
            minus:add(c2)
            connection:add(minus) -- D*2            
            local similarityC=nn.Concat(1) -- multi similarity criteria     
            local s1=nn.Sequential()
            s1:add(nn.SplitTable(2))
            s1:add(nn.PairwiseDistance(2)) -- scalar
            local s2=nn.Sequential()          
            if 1 < 3 then
              s2:add(nn.SplitTable(2))
            else
              s2:add(nn.Transpose({1,2})) -- D*2 -> 2*D
              s2:add(nn.SoftMax())
              s2:add(nn.SplitTable(1))                    
            end             
            s2:add(nn.CsDis()) -- scalar
            local s3=nn.Sequential()
            s3:add(nn.SplitTable(2))
            s3:add(nn.CSubTable()) -- linear
            s3:add(nn.Abs()) -- linear            
            similarityC:add(s1)
            similarityC:add(s2)         
            similarityC:add(s3)         
            connection:add(similarityC) -- scalar                     
            d:add(connection)
            --end
          end   
        end
      end 
                  
      for i=1,NumFilter do
        for j=1,3 do 
          local connection = nn.Sequential()
          connection:add(nn.Select(1,i)) -- == 2items
          connection:add(nn.Reshape(2*separator,1)) --2items*1 here         
          local minus=nn.Concat(2)
          local c1=nn.Sequential()
          local c2=nn.Sequential()
          if j == 1 then 
            c1:add(nn.Narrow(1,1,ngram+1)) -- first half (items/3)*1
            c2:add(nn.Narrow(1,separator+1,ngram+1)) -- first half (items/3)*1
          elseif j == 2 then
            c1:add(nn.Narrow(1,ngram+2,ngram+1)) -- 
            c2:add(nn.Narrow(1,separator+ngram+2,ngram+1)) 
          else
            c1:add(nn.Narrow(1,2*(ngram+1)+1,ngram+1)) 
            c2:add(nn.Narrow(1,separator+2*(ngram+1)+1,ngram+1)) --each is ngram+1 portion (max or min or mean)
          end           
          
          minus:add(c1)
          minus:add(c2)
          connection:add(minus) -- (items/3)*2          
          local similarityC=nn.Concat(1)  
          local s1=nn.Sequential()
          s1:add(nn.SplitTable(2))
          s1:add(nn.PairwiseDistance(2)) -- scalar
          local s2=nn.Sequential()          
          if 1 >= 2 then
            s2:add(nn.Transpose({1,2})) -- (items/3)*2 -> 2*(items/3)
            s2:add(nn.SoftMax()) --for softmax have to do transpose from (item/3)*2 -> 2*(item/3)
            s2:add(nn.SplitTable(1)) --softmax only works on row            
          else                                        
            s2:add(nn.SplitTable(2)) --(items/3)*2
          end
          s2:add(nn.CsDis()) -- scalar
          --local s3=nn.Sequential()
          --s3:add(nn.SplitTable(2))
          --s3:add(nn.CSubTable()) -- linear
          --s3:add(nn.Abs()) -- linear            
          similarityC:add(s1)
          similarityC:add(s2)         
          --similarityC:add(s3)
          connection:add(similarityC) -- scalar                     
          d:add(connection)       
        end
      end     
      
      for i=items+1,separator do
        local connection = nn.Sequential()
        local minus=nn.Concat(2)
        local c1=nn.Sequential()
        local c2=nn.Sequential()
        c1:add(nn.Select(2,i)) -- == D, not D*1
        c1:add(nn.Reshape(NumFilter,1)) --D*1 here
        c2:add(nn.Select(2,separator+i))
        c2:add(nn.Reshape(NumFilter,1))
        minus:add(c1)
        minus:add(c2)
        connection:add(minus) -- D*2            
        local similarityC=nn.Concat(1)      
        local s1=nn.Sequential()
        s1:add(nn.SplitTable(2))
        s1:add(nn.PairwiseDistance(2)) -- scalar
        local s2=nn.Sequential()          
        if 1 < 3 then
          s2:add(nn.SplitTable(2))
        else
          s2:add(nn.Transpose({1,2})) 
          s2:add(nn.SoftMax())
          s2:add(nn.SplitTable(1))                    
        end             
        s2:add(nn.CsDis()) -- scalar
        local s3=nn.Sequential()
        s3:add(nn.SplitTable(2))
        s3:add(nn.CSubTable()) -- linear
        s3:add(nn.Abs()) -- linear            
        similarityC:add(s1)
        similarityC:add(s2)         
        similarityC:add(s3)         
        connection:add(similarityC) -- scalar                     
        d:add(connection)   
      end
      
      deepQuery:add(d)      
      return deepQuery  
    end
end

print('createModel')

  
  
  
  --dofile 'models.lua'     -- Opens the named file and executes its contents as a Lua chunk --
  print('<model> creating a fresh model')
  
  -- Type of model; Size of vocabulary; Number of output classes
  local modelName = 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt'
  print(modelName)
  self.ngram = 3
  self.length = self.emb_dim
  self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram)  
  self.softMaxC = self:ClassifierOOne()
  print('self:')
  print(self)
  ----------------------------------------
  local modules = nn.Parallel()
    :add(self.convModel) 
    :add(self.softMaxC) 
  self.params, self.grad_params = modules:getParameters()
  print('self.params: ' .. self.params:norm())
  -- print('self.grad_params: ' .. self.grad_params)
  -- print(self.params:norm())
  -- print(self.convModel:parameters()[1][1]:norm())
  -- print(self.softMaxC:parameters()[1][1]:norm())
end

function Conv:ClassifierOOne()
  local maxMinMean = 3
  local separator = (maxMinMean+1)*self.mem_dim
  local modelQ1 = nn.Sequential() 
  local ngram = self.ngram
  local items = (ngram+1)*3     
  -- local items = (ngram+1) -- no Min and Mean
  local NumFilter = self.length -- 300
  local conceptFNum = 20          -- the number of filter is 20 --
  inputNum = 2*items*items/3+NumFilter*items*items/3+6*NumFilter+(2+NumFilter)*2*ngram*conceptFNum --PoinPercpt model!
  modelQ1:add(nn.Linear(inputNum, self.sim_nhidden))
  modelQ1:add(nn.Tanh())  
  modelQ1:add(nn.Linear(self.sim_nhidden, self.num_classes))
  modelQ1:add(nn.LogSoftMax())  
  return modelQ1
end

function Conv:trainCombineOnly(dataset)
  -- local classes = {1,2}
  -- local confusion = optim.ConfusionMatrix(classes)
  -- confusion:zero()
  train_looss = 0.0
   
  local indices = torch.randperm(dataset.size)  -- 打乱顺序--
  local zeros = torch.zeros(self.mem_dim)       -- 初始化一个self.mem_dim大小的数组，数组元素为0 --
  for i = 1, dataset.size, self.batch_size do   -- for start, end, step do statement end
    -- if i%10 == 1 then
    --      xlua.progress(i, dataset.size)
    -- end
    print("dataset.size: ")
    print(dataset.size)
    print("self.batch_size: ")
    print(self.batch_size)
    print("batch_size: ")
    print(batch_size)
    local batch_size = 1 -- math.min(i + self.batch_size - 1, dataset.size) - i + 1
    -- get target distributions for batch
    print("batch_size2:")
    print(batch_size)
    print('self.num_classes: ')
    print(self.num_classes)
    local targets = torch.zeros(batch_size, self.num_classes)
    print("targets: ")
    print(targets)
    for j = 1, batch_size do
    print('batch size1: ')
    print(batch_size)
      local sim  = -0.1
      if self.task == 'sic' or self.task == 'vid' then
--        print("i: " .. i)
--        print("j: " .. j)
--        print('i + j - 1: ' .. i + j - 1)
--        print("self.num_classes - 1: " .. self.num_classes - 1)
--        print("dataset.labels[indices[i + j - 1]]: " .. dataset.labels[indices[i + j - 1]])
--        print("sim: ")
        sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
        print(sim)
      elseif self.task == 'pit' then
        sim = dataset.labels[indices[i + j - 1]] + 1 
      elseif self.task == 'others' then
        sim = dataset.labels[indices[i + j - 1]] + 1 
      else
        error("not possible!")
      end
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil - sim
        targets[{j, ceil}] = sim - floor
      end
    end
    
    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        print('linputs: ')
        print(linputs)
        
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()
        print('rinputs: ')
        print(rinputs)
        
        local part2 = self.convModel:forward({linputs, rinputs})
        local output = self.softMaxC:forward(part2)
--        print('linputs: ')
--        print(linputs)
--        print('rinputs: ')
--        print(rinputs)
--        print('part2: ')
--        print(part2)
--        print('output: ')
--        print(output)
        
        loss = self.criterion:forward(output, targets[1])
        print('targets[1]: ')
        print(targets[1])
        print("loss: ")
        print(loss)
        train_looss = loss + train_looss
        local sim_grad = self.criterion:backward(output, targets[1])
        local gErrorFromClassifier = self.softMaxC:backward(part2, sim_grad)
        self.convModel:backward({linputs, rinputs}, gErrorFromClassifier)
      end
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
    -- train_looss = train_looss + fs[#fs]
  end
  print('Loss: ' .. train_looss)
end

-- Predict the similarity of a sentence pair.
function Conv:predictCombination(lsent, rsent)
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()

  local part2 = self.convModel:forward({linputs, rinputs})
  print('part2: ')
  print(part2)
  local output = self.softMaxC:forward(part2)         -- print output for val --
  print("output: ")
  print(output)
  print("output:exp(): ")
  print(output:exp())
  print("torch.range(1,5,1): ")
  print(torch.range(1,5,1))
  print("torch.range(1, 5, 1):dot(output:exp()): ")
  print(torch.range(1, 5, 1):dot(output:exp()))
  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'pit' then                    -- add pit task --
    val = torch.range(0, 1):dot(output:exp())     
  else
    error("not possible task")
  end
  print('val:')
  print(val)
  return val
end

-- Produce similarity predictions for each sentence pair in the data set.
function Conv:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
  end
  return predictions
end

function Conv:print_config()
  local num_params = self.params:nElement()

  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('LSTM memory dim: ' .. self.mem_dim)
  print('regularization strength: ' .. self.reg)
  print('minibatch size: ' .. self.batch_size)
  print('learning rate: ' .. self.learning_rate)
  print('LSTM structure: ' .. self.structure)
  print('LSTM layers: ' .. self.num_layers)
  print('sim module hidden dim: ' .. self.sim_nhidden)
end

function Conv:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs:float(),
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    mem_dim       = self.mem_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end
