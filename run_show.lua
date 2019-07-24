require 'image'
require 'dataset_load'
require 'network_model'
require 'camera_model'
require 'warping_model'
require 'pointcloud_model'
local prepro = require 'prepro'
local util = require 'util'

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:text("Options")
cmd:option("-dataset_file", "dataset_test_5m1m.txt", "dataset file")
cmd:option("-input_width", 320, "input width")
cmd:option("-input_height", 240, "input height")
cmd:option("-input_channels", 3, "input channels")
cmd:option('-gpu', 1, 'GPU to use. 0 = no GPU')
cmd:option('-batch_size', 1, 'Batch size')
cmd:option('-display', 0,'Display images')
cmd:option('-resume','train_model.t7','Resume from a model')

local opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
torch.manualSeed(123)

if opt.gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cudnn.benchmark=true
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeed(123)
end

local model = DeltaOdom(opt.gpu)

local network
if opt.resume ~= '' then
  network = model:load_network(opt.resume)
else
  network = model:build_network(opt.input_height, opt.input_width)
end

if opt.gpu>0 then
  network=network:cuda()
end

network:evaluate()

local warping_model = build_warping_model(opt.input_height, opt.input_width)
local pointcloud_model = build_pointcloud_model(opt.input_height, opt.input_width)
if opt.gpu>0 then
  warping_model=warping_model:cuda()
  pointcloud_model=pointcloud_model:cuda()
end

local data = load_datasets(opt.dataset_file)
local data_size = #data

local pose_error = {}
for sample = 1, #data, opt.batch_size do
  local curr_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
  local base_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
  local depth_data = torch.Tensor(opt.batch_size,1,opt.input_height,opt.input_width)
  local pose_data  = torch.Tensor(opt.batch_size,3)
  local camera_data= torch.Tensor(opt.batch_size,4)

  for j=1, opt.batch_size do
    local curr_frame  = image.load(data[sample][4],opt.input_channels,'float')
    local base_frame  = image.load(data[sample][5],opt.input_channels,'float')

    local image_height, image_width = base_frame:size(2), base_frame:size(3)
    local crop_offset  = prepro.random_crop(opt.input_height, opt.input_width, image_height, image_width)
    local camera_model = CameraIntrinsics(image_height, image_width, data[sample][2])

    local depth_frame = image.load(data[sample][3],1,'byte'):float()
    depth_frame  = torch.clamp(depth_frame,0.1,math.huge)
    depth_frame  = torch.pow(depth_frame,-1.0)
    depth_frame  = depth_frame * camera_model:focalLength() * camera_model:baseline()

    camera_data[j] = torch.Tensor(camera_model:intrinsics(crop_offset))
    pose_data[j]   = torch.Tensor(data[sample][1])
    depth_data[j]  = prepro.crop_image(depth_frame, crop_offset, opt.input_height, opt.input_width)
    curr_data[j]   = prepro.crop_image(curr_frame, crop_offset, opt.input_height, opt.input_width)
    base_data[j]   = prepro.crop_image(base_frame, crop_offset, opt.input_height, opt.input_width)
  end  

  if opt.gpu>0 then
    camera_data = camera_data:cuda()
    pose_data   = pose_data:cuda()
    depth_data  = depth_data:cuda()
    curr_data   = curr_data:cuda()
    base_data   = base_data:cuda()
  end

  local inputs = {curr_data, base_data, depth_data, camera_data}

  local pred_cloud = network:forward(inputs):clone()
  local pred_warp = warping_model:forward({base_data,pred_cloud,camera_data}):clone()

  local true_cloud = pointcloud_model:forward({pose_data,depth_data,camera_data}):clone()
  local true_warp = warping_model:forward({base_data,true_cloud,camera_data}):clone()

  local pred_pose = model:get_delta_pose_pred():clone()
  local erro_pose = util.distance(pred_pose,pose_data)

  print(string.format('mean error xyz: %.4f', erro_pose))
  
  pose_error[#pose_error+1] = erro_pose

  if opt.display==1 then
    --win1 = image.display{image = base_data[1], zoom = 1, win=win1, legend = 'keyframe'}
    win2 = image.display{image = curr_data[1], zoom = 1, win=win2, legend = 'current'}
    win3 = image.display{image = true_warp[1], zoom = 1, win=win3, legend = 'true_warped'}
    --win4 = image.display{image = pred_warp[1], zoom = 1, win=win4, legend = 'pred_warped'}
  end
  --[[
  if err_xyz > 4.5 then
    print(data[sample][4])
    print(data[sample][5])
    local key = io.read()
  end
  --]]
end
local mean_error = torch.mean(torch.Tensor(pose_error))
local stdv_error = torch.std(torch.Tensor(pose_error))
print(string.format('mean error: %.2fm std dev: %.2fm', mean_error, stdv_error))
