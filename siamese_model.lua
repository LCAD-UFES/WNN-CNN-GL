require 'nn'

function build_network_model(gpu)
  local base_encoder = nn.Sequential()
  base_encoder:add(nn.SpatialConvolution( 3,  64, 3, 3, 2, 2, 1, 1))
  base_encoder:add(nn.PReLU()) 
  
  base_encoder:add(nn.SpatialConvolution(64,  64, 3, 3, 2, 2, 1, 1)) 
  base_encoder:add(nn.PReLU())
  
  base_encoder:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, 1, 1)) 
  base_encoder:add(nn.PReLU())
  
  local base_encoder_init = require('weight-init')(base_encoder, 'MSRinit')
  local base_encoder_clone = base_encoder_init:clone()
  base_encoder_clone:share(base_encoder_init, 'weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std', 'running_var')
  
  local siamese_encoder = nn.ParallelTable()
  siamese_encoder:add(base_encoder_init)
  siamese_encoder:add(base_encoder_clone) 
  
  local top_encoder = nn.Sequential()
  top_encoder:add(nn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  top_encoder:add(nn.PReLU()) 
  
  top_encoder:add(nn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1)) 
  top_encoder:add(nn.PReLU())
  
  top_encoder:add(nn.SpatialConvolution(256, 512, 3, 3, 2, 2, 1, 1)) 
  top_encoder:add(nn.PReLU())
  
  top_encoder:add(nn.SpatialConvolution(512, 1024, 3, 3, 2, 2, 1, 1)) 
  top_encoder:add(nn.PReLU())
  
  top_encoder:add(nn.SpatialConvolution(1024, 4096, 3, 3, 2, 2, 1, 1)) 
  top_encoder:add(nn.Dropout(0.5))
  top_encoder:add(nn.PReLU())
  
  top_encoder:add(nn.SpatialConvolution(4096, 4096, 2, 1, 1, 1, 1, 1)) 
  top_encoder:add(nn.Dropout(0.5))
  top_encoder:add(nn.PReLU())
  
  local top_encoder_init = require('weight-init')(top_encoder, 'MSRinit')
  
  local pred_layer = nn.SpatialConvolution(4096, 3, 3, 3, 1, 1, 0, 0)
  pred_layer.weight:zero()
  pred_layer.bias:zero()
  
  local model = nn.Sequential()
  model:add(siamese_encoder)
  model:add(nn.JoinTable(2))
  model:add(top_encoder_init)
  model:add(pred_layer)
  model:add(nn.View(-1,3))
  
  local input = torch.Tensor(2, 3, 240, 320)
  
  if gpu>0 then
    model = model:cuda()
    input = input:cuda()
    cudnn.convert(model, cudnn)
    local optnet = require 'optnet'
    optnet.optimizeMemory(model, {input,input}, {inplace=true, mode='training'})
  end

  return model
end

--[[
require 'cudnn'
local model = build_network_model(0)

local input = torch.Tensor(2, 3, 240, 320)

--input = input:cuda()
--model = model:cuda()

local output = model:forward({input,input})

print(model)
print(output:size())

params, grad_params = model:getParameters()
print('Number of parameters ' .. params:nElement())

--]]