require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'network_model'
-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:text("Options")
cmd:option('-gpu', 1, 'GPU to use. 0 = no GPU')
cmd:option('-input','best_model.t7','input model')
cmd:option('-output','slim_model.t7','output model')

local opt = cmd:parse(arg)

local model = DeltaOdom()

local network
if opt.input ~= '' then
  network = model:load_network(opt.input, true)
end

if opt.gpu==0 then
  cudnn.convert(network, nn)
  network = network:float()
end

torch.save(opt.output, network)