local util = torch.class('pit.util')

function pit.header(msg)
  print(string.rep('-', 100))
  print(msg)
  print(string.rep('-', 100))
end

function pit.pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

function pit.sentSplit(sent, tag)
  local tokens = {}
  while (true) do
    local pos = string.find(sent, tag)
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

function pit.readEmb(voc, emb)
  local v = pit.vocab(voc)
  local e = torch.load(emb)
  return v, e
end

function pit.readSent(path, vocab)
  local sents = {}
  local file = io.open(path, 'r')
  local line  
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = pit.sentSplit(line, " ")
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

function pit.readData(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsent = pit.readSent(dir .. 'ls.txt', vocab)
  dataset.rsent = pit.readSent(dir .. 'rs.txt', vocab)
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