local vocab = torch.class('pit.vocab')

function vocab:__init(path)
  self.size = 0
  self._index = {}
  self._tokens = {}

  local file = io.open(path)
  while true do
    local line = file:read()
    if line == nil then break end
    self.size = self.size + 1
    self._tokens[self.size] = line
    self._index[line] = self.size
  end
  file:close()

  local unks = {'<unk>', '<UNK>', 'UUUNKKK'}
  for _, tok in pairs(unks) do
    self.unk_index = self.unk_index or self._index[tok]
    if self.unk_index ~= nil then
      self.unk_token = tok
      break
    end
  end

  local starts = {'<s>', '<S>'}
  for _, tok in pairs(starts) do
    self.start_index = self.start_index or self._index[tok]
    if self.start_index ~= nil then
      self.start_token = tok
      break
    end
  end

  local ends = {'</s>', '</S>'}
  for _, tok in pairs(ends) do
    self.end_index = self.end_index or self._index[tok]
    if self.end_index ~= nil then
      self.end_token = tok
      break
    end
  end
end

function vocab:contains(w)
  if not self._index[w] then return false end
  return true
end

function vocab:add(w)
  if self._index[w] ~= nil then
    return self._index[w]
  end
  self.size = self.size + 1
  self._tokens[self.size] = w
  self._index[w] = self.size
  return self.size
end

function vocab:index(w)
  local index = self._index[w]
  if index == nil then
    if self.unk_index == nil then
      error('Token not in vocabulary and no UNK token defined: ' .. w)
    end
    return self.unk_index
  end
  return index
end

function vocab:token(i)
  if i < 1 or i > self.size then
    error('Index ' .. i .. ' out of bounds')
  end
  return self._tokens[i]
end

function vocab:map(tokens)
  local len = #tokens
  local output = torch.IntTensor(len)
  for i = 1, len do
    output[i] = self:index(tokens[i])
  end
  return output
end

function vocab:add_unk_token()
  if self.unk_token ~= nil then return end
  self.unk_index = self:add('<unk>')
end

function vocab:add_start_token()
  if self.start_token ~= nil then return end
  self.start_index = self:add('<s>')
end

function vocab:add_end_token()
  if self.end_token ~= nil then return end
  self.end_index = self:add('</s>')
end
