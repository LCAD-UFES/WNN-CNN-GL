require 'nn'
require 'torch'
require 'network_model'
require 'camera_params'

function join(list, sep)
   local sep = sep or ' '
   return table.concat(list, sep)
end

function show_layer_sizes(input, m)
    local m = m:clone()
    local output = m:forward(input)

    function rec(m, layer)
       local count = 1
       for k, v in pairs(m.modules) do
          -- Print layer's output size:
          if v['output'] ~= nil then
             local result = string.rep('-', layer) .. string.format(' (%s) ', count)
             if torch.type(v['output']) == 'table' then
                for _, t in pairs(v['output']) do
                   local size_str = join(torch.totable(t:size()), 'x')
                   result = result .. size_str .. ' '
                end
             else
                local size_str = join(torch.totable(v['output']:size()), 'x')
                result = result .. size_str
             end
             print(result)
          end

          -- Recurse into layer's submodules:
          if v['modules'] ~= nil then
             rec(v, layer+1)
          end
          count = count + 1
       end
    end
    rec(m, 1)
end

do
-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text("Arguments")
  cmd:text("Options")
  cmd:option("-camera_file", "/dados/ICL-NUIM/camerapar-iclnuim.txt", "camera intrinsics params file")
  cmd:option("-input_width", 640/2, "input width")
  cmd:option("-input_height", 480/2, "input height")
  cmd:option('-batch_size', 1, 'Batch size')

  local opt = cmd:parse(arg)

  local model = DeltaOdom()

  local camera_params = load_camera_intrinsics(opt.camera_file, opt.input_height, opt.input_width, opt.input_height, opt.input_width)

  local network = model:build_network(camera_params)

  local curr_data = torch.Tensor(opt.batch_size,3,opt.input_height,opt.input_width)
  local base_data = torch.Tensor(opt.batch_size,3,opt.input_height,opt.input_width)
  local depth_data= torch.Tensor(opt.batch_size,1,opt.input_height,opt.input_width)
  local pose_data = torch.Tensor(opt.batch_size,6)

  local input = {curr_data, base_data, depth_data, pose_data}

  local output = network:forward{input,input}

  local siamese = model:get_siamese_model()

  --print(siamese.modules)
  
  show_layer_sizes(input, siamese)

end
