require 'nn'
--require 'inn' --luarocks install inn
require 'image'
require 'torch'
require 'dataset_load'
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
cmd:option('-resume','slim_model.t7','Resume from a model')

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
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeed(123)
end

local network
if opt.resume ~= '' then
  network = torch.load(opt.resume)
end
print(network) --print network archtecture
if opt.gpu>0 then
  network=network:cuda()
else
  network=network:float()
end

local data = load_datasets(opt.dataset_file, true)

local pose_error = {}
local timer = torch.Timer()
for sample = 1, #data, opt.batch_size do
  local curr_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
  local base_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
  local pose_data  = torch.Tensor(opt.batch_size,3)
  for j=1,opt.batch_size do
    local curr_frame  = image.load(data[sample][4],opt.input_channels,'float')
    local base_frame  = image.load(data[sample][5],opt.input_channels,'float')
    local image_height, image_width = base_frame:size(2), base_frame:size(3)
    local crop_offset  = prepro.center_crop(opt.input_height, opt.input_width, image_height, image_width)
    curr_data[j]   = prepro.crop_image(curr_frame, crop_offset, opt.input_height, opt.input_width)
    base_data[j]   = prepro.crop_image(base_frame, crop_offset, opt.input_height, opt.input_width)
    pose_data[j]   = torch.Tensor(data[sample][1])
  end
 
  if opt.gpu>0 then
    base_data=base_data:cuda()
    curr_data=curr_data:cuda()
    pose_data=pose_data:cuda()
  end

  timer:reset()
  local output = network:forward({curr_data, base_data})
  local elapsed_time = timer:time().real
  pose_error[#pose_error+1] = util.distance(output,pose_data)
  --print(string.format('error: %.4f time: %.1f Hz', pose_error[#pose_error], 1.0/elapsed_time))

  if opt.display==1 then
    win1 = image.display{image = base_data[1], zoom = 1, win=win1, legend = 'keyframe'}
    win2 = image.display{image = curr_data[1], zoom = 1, win=win2, legend = 'current'}
  end
  local mean_pred_pose = torch.mean(output,1)
  local mean_true_pose = torch.mean(pose_data,1)
  if pose_error[#pose_error] > 0.5 then
    --print(string.format('*pred pose: %.1f° %.2fm %.2fm', math.deg(mean_pred_pose[1][2]), mean_pred_pose[1][4], mean_pred_pose[1][6]))
    --print(string.format('*true pose: %.1f° %.2fm %.2fm', math.deg(mean_true_pose[1][2]), mean_true_pose[1][4], mean_true_pose[1][6]))
    print(string.format('*pred pose: %.1f° %.2fm %.2fm', math.deg(mean_pred_pose[1][1]), mean_pred_pose[1][2], mean_pred_pose[1][3]))
    print(string.format('*true pose: %.1f° %.2fm %.2fm', math.deg(mean_true_pose[1][1]), mean_true_pose[1][2], mean_true_pose[1][3]))
    --local key = io.read()
  else
    --print(string.format('pred pose: %.1f° %.2fm %.2fm', math.deg(mean_pred_pose[1][2]), mean_pred_pose[1][4], mean_pred_pose[1][6]))
    --print(string.format('true pose: %.1f° %.2fm %.2fm', math.deg(mean_true_pose[1][2]), mean_true_pose[1][4], mean_true_pose[1][6]))
    print(string.format('pred pose: %.1f° %.2fm %.2fm', math.deg(mean_pred_pose[1][1]), mean_pred_pose[1][2], mean_pred_pose[1][3]))
    print(string.format('true pose: %.1f° %.2fm %.2fm', math.deg(mean_true_pose[1][1]), mean_true_pose[1][2], mean_true_pose[1][3]))
  end
end
local mean_error = torch.mean(torch.Tensor(pose_error))
local stdv_error = torch.std(torch.Tensor(pose_error))
print("*error greater than 0.5m") 
print(string.format('mean error: %.2fm std dev: %.2fm', mean_error, stdv_error))
return mean_error