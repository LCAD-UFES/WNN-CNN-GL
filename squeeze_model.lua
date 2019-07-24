-- implementation of squeezenet proposed in: http://arxiv.org/abs/1602.07360
require 'nn'

local function fire(ch, s1, e1, e3)
  local net = nn.Sequential()
  net:add(nn.SpatialConvolution(ch, s1, 1, 1))
  net:add(nn.ReLU(true))
  local exp = nn.Concat(2)
  exp:add(nn.SpatialConvolution(s1, e1, 1, 1))
  exp:add(nn.SpatialConvolution(s1, e3, 3, 3, 1, 1, 1, 1))
  net:add(exp)
  net:add(nn.ReLU(true))
  return net
end


local function bypass(net)
  local cat = nn.ConcatTable()
  cat:add(net)
  cat:add(nn.Identity())
  local seq = nn.Sequential()
  seq:add(cat)
  seq:add(nn.CAddTable(true))
  return seq
end

function build_network_model(gpu)
  local net = nn.Sequential()
  net:add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(2)))
  net:add(nn.JoinTable(2))
  net:add(nn.SpatialConvolution(6, 96, 9, 9, 3, 3, 0, 0)) -- conv1
  net:add(nn.ReLU(true))
  net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  net:add(fire(96, 16, 64, 64))  --fire2
  net:add(bypass(fire(128, 16, 64, 64)))  --fire3
  net:add(fire(128, 32, 128, 128))  --fire4
  net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  net:add(bypass(fire(256, 32, 128, 128)))  --fire5
  net:add(fire(256, 48, 192, 192))  --fire6
  net:add(bypass(fire(384, 48, 192, 192)))  --fire7
  net:add(fire(384, 64, 256, 256))  --fire8
  net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  net:add(bypass(fire(512, 64, 256, 256)))  --fire9
  net:add(nn.Dropout(0.5))
  net:add(nn.SpatialConvolution(512, 64, 3, 3, 2, 2, 0, 0)) --conv10
  net:add(nn.ReLU(true))

  local net_init = require('weight-init')(net, 'kaiming')
  
  local pred_layer = nn.SpatialConvolution(64, 3, 3, 3, 3, 2, 0, 0)
  pred_layer.weight:zero()
  pred_layer.bias:zero()

  local model = nn.Sequential()
  model:add(net_init)
  model:add(pred_layer)
  
  model:add(nn.View(-1,3))
  if gpu>0 then
    cudnn.convert(model, cudnn)
  end
  return model
end

--[[
require 'cudnn'
local model = build_squeeze_model(1)

local input = torch.Tensor(2, 3, 240, 320)

input = input:cuda()
model = model:cuda()

local output = model:forward({input,input})

print(model)
print(output:size())
--]]