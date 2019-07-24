require 'torch'
require 'network_model'
require 'camera_params'

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:text("Options")
cmd:option("-camera_file", "/dados/ICL-NUIM/camerapar-iclnuim.txt", "camera intrinsics params file")
cmd:option("-input_width", 320, "input width")
cmd:option("-input_height", 180, "input height")
cmd:option('-batch_size', 1, 'Batch size')

local opt = cmd:parse(arg)

local model = DeltaOdom()

local camera_params = load_camera_intrinsics(opt.camera_file, opt.input_height, opt.input_width, opt.input_height, opt.input_width)

local network = model:build_network(camera_params)

local curr_data = torch.Tensor(opt.batch_size,3,opt.input_height,opt.input_width)
local base_data = torch.Tensor(opt.batch_size,3,opt.input_height,opt.input_width)
local depth_data = torch.Tensor(opt.batch_size,1,opt.input_height,opt.input_width)
local pose_data  = torch.Tensor(opt.batch_size,6)

local input = {curr_data, base_data, depth_data, pose_data}

local output = network:forward{input,input}

local siamese = model:get_siamese_model()

local generateGraph = require 'optnet.graphgen'

  -- visual properties of the generated graph
-- follows graphviz attributes
graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}
local g =  generateGraph(siamese, input, graphOpts)

graph.dot(g, 'Siamese', 'graph')
--run dot < graph.dot -T pdf -o graph.pdf