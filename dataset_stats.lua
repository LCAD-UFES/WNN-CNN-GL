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
cmd:option("-camera_file", "camera3.txt", "camera intrinsics params file")
cmd:option("-dataset_file", "dataset_list.txt", "dataset file")
cmd:option("-image_width", 640, "image width")
cmd:option("-image_height", 380, "image height")
cmd:option("-input_width", 320, "input width")
cmd:option("-input_height", 240, "input height")
cmd:option("-input_channels", 3, "input channels")
cmd:option('-gpu', 1, 'GPU to use. 0 = no GPU')
cmd:option('-batch_size',100, 'Batch size')

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

local camera_model = CameraIntrinsics(opt.camera_file, 480, 640)

local data = load_datasets(opt.dataset_file, false)
local data_size = #data

local base_mean = {0,0,0}
local base_std = {0,0,0}
local curr_mean = {0,0,0}
local curr_std = {0,0,0}
print("Reading images")
for batch=1, data_size, opt.batch_size do
  xlua.progress(batch, data_size) 

  local batch_size = opt.batch_size

  if batch+batch_size > data_size then
    batch_size = data_size % batch_size
  end
  
  local curr_data  = torch.Tensor(batch_size,opt.input_channels,opt.image_height,opt.image_width)
  local base_data  = torch.Tensor(batch_size,opt.input_channels,opt.image_height,opt.image_width)

  for j=0, batch_size-1 do
    local curr_frame  = image.load(data[batch+j][1],opt.input_channels,'byte'):float()
    local base_frame  = image.load(data[batch+j][2],opt.input_channels,'byte'):float()
    curr_data[j+1] = curr_frame
    base_data[j+1] = base_frame
  end

  if opt.gpu>0 then
    curr_data = curr_data:cuda()
    base_data = base_data:cuda()
  end

  local crop_offset = prepro.center_crop(batch_size, opt.input_height, opt.input_width, opt.image_height, opt.image_width)
  curr_data = prepro.crop_image(curr_data, crop_offset, opt.input_height, opt.input_width)
  base_data = prepro.crop_image(base_data, crop_offset, opt.input_height, opt.input_width)

  for j=1,3 do
    base_mean[j] = base_mean[j] + base_data:select(2,j):mean()
    base_std[j] = base_std[j] + base_data:select(2,j):std()
    curr_mean[j] = curr_mean[j] + curr_data:select(2,j):mean()
    curr_std[j] = curr_std[j] + curr_data:select(2,j):std()
  end

end

print("Saving stats")
for j=1,3 do
  base_mean[j] = base_mean[j] * opt.batch_size / data_size
  base_std[j] = base_std[j] * opt.batch_size / data_size
  curr_mean[j] = curr_mean[j] * opt.batch_size / data_size
  curr_std[j] = curr_std[j] * opt.batch_size / data_size
end

local base_stats = {["mean"] = base_mean, ["std"] = base_std}
local curr_stats = {["mean"] = curr_mean, ["std"] = curr_std}

print(base_stats.mean[1],base_stats.mean[2],base_stats.mean[3],base_stats.std[1],base_stats.std[2],base_stats.std[3])
print(curr_stats.mean[1],curr_stats.mean[2],curr_stats.mean[3],curr_stats.std[1],curr_stats.std[2],curr_stats.std[3])

torch.save("base_stats.t7", base_stats)
torch.save("curr_stats.t7", curr_stats)

print("Done")