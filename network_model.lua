require 'nn'
require 'torch'
require 'pointcloud_model'
require 'siamese_model'

do

  local DeltaOdom = torch.class('DeltaOdom')

  function DeltaOdom:__init(gpu)
    self.gpu = gpu
  end

  function DeltaOdom:load_network(filename, inference)
    self.network = torch.load(filename)

    inference = inference or false
    if inference == true then
      --returns the siamese model for inference
      local posenet = self.network:get(1):get(1)
      self.network = posenet:clearState()
    end

    return self.network
  end

  function DeltaOdom:build_network(height, width, filename)
    filename = filename or nil
    local warpnet_model = build_pointcloud_model(height, width)
    local posenet_model
    if filename == nil then
      posenet_model = build_network_model(self.gpu)
    else
      local clone = load_network_model_pretrain(self.gpu, filename)
      posenet_model = clone:clearState():clone() -- this copies all the parameters and gradParameters
      posenet_model:share(clone,'weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std', 'running_var') -- this deletes and replaces them
    end

    local posenet_input = nn.ConcatTable()
    posenet_input:add(nn.SelectTable(1)) --Live Image
    posenet_input:add(nn.SelectTable(2)) --Ref. Image
    local posenet = nn.Sequential()
    posenet:add(posenet_input)
    posenet:add(posenet_model)

    local warpnet_input = nn.ConcatTable()
    warpnet_input:add(posenet)            --Delta pose
    warpnet_input:add(nn.SelectTable(3))  --Depth data
    warpnet_input:add(nn.SelectTable(4))  --Intrinsics
    local warpnet = nn.Sequential()
    warpnet:add(warpnet_input)
    warpnet:add(warpnet_model)
    self.network = warpnet
    return self.network
  end

  function DeltaOdom:get_warping_output_pred()
    return self.network:get(2):get(2).output:clone()
  end

  function DeltaOdom:get_delta_pose_pred()
    return self.network:get(1):get(1).output:clone()
  end

  function DeltaOdom:get_delta_pose_grad()
    -- returns gradient of TransformationMatrix3x4SE2 with respect to its input (posenet)
    return self.network:get(2):get(1):get(1):get(2).gradInput:clone()
  end

end
