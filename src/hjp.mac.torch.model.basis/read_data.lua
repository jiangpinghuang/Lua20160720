--[[
  Functions for loading data from disk.
--]]

function lua_string_split(str, split_char)
    local sub_str_tab = {};
    while (true) do
        local pos = string.find(str, split_char);
        if (not pos) then
            sub_str_tab[#sub_str_tab + 1] = str;
            break;
        end
        local sub_str = string.sub(str, 1, pos - 1);
        sub_str_tab[#sub_str_tab + 1] = sub_str;
        str = string.sub(str, pos + 1, #str);
    end
    for i = 1, #sub_str_tab do
      print(sub_str_tab[i])
    --print(sub_str_tab)
    end
    return sub_str_tab;
end

function similarityMeasure.read_embedding(vocab_path, emb_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function similarityMeasure.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    --local tokens = stringx.split(line)
    local tokens = lua_string_split(line, " ")
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    --print('sent: ')
    --print(sent)
    --print("#sentences: ")
    --print(#sentences)
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsents = similarityMeasure.read_sentences(dir .. 'ls.txt', vocab)
  dataset.rsents = similarityMeasure.read_sentences(dir .. 'rs.txt', vocab)
  dataset.size = #dataset.lsents
  print('dir: ')
  print(dir)
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    if task == 'sic' then
      --dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
      print('sim_file:readDouble():')
      local num_sim = sim_file:readDouble()
      
      print(num_sim)
      print('sim_file:readDouble() - 1:')
      print(num_sim - 1)
      print('dataset.labels[i]: ')
      print(0.25 * (num_sim - 1)) 
      dataset.labels[i] = 0.25 * (num_sim - 1)
    elseif task == 'vid' then
      dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
    else
      dataset.labels[i] = (sim_file:readDouble()) -- twi and msp
    end
  end
  id_file:close()
  sim_file:close()
  return dataset
end

