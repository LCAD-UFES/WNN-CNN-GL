require 'optim' 
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
cmd:option("-dataset_file", "dataset_train_5m1m.txt", "dataset file")
cmd:option("-dataset_test", "dataset_test_5m1m.txt", "dataset file for validation")
cmd:option("-input_width", 320, "input width")
cmd:option("-input_height", 240, "input height")
cmd:option("-input_channels", 3, "input channels")
cmd:option('-gpu', 1, 'GPU to use. 0 = no GPU')
cmd:option('-batch_size',48, 'Batch size')
cmd:option('-max_iter',5e4,'number of iterations')
cmd:option('-learning_rate', 1e-4,'learning rate')
cmd:option('-verbose', 1,'Print messages interval')
cmd:option('-display', 0,'Display images interval')
cmd:option('-save', 200,'Save training interval')
cmd:option('-resume','','Resume from a model')

local opt = cmd:parse(arg)

local hyper_params = {
  learningRate = opt.learning_rate,
  learningRateDecay = 0, --set below
  weightDecay = 0, --0.0005,
  epsilon = 1e-8,
  beta1 = 0.9,
  beta2 = 0.999
}

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
  hyper_params = torch.load('train_state.t7')
else
  network = model:build_network(opt.input_height, opt.input_width)--, 'pretrain_model.t7')
end

if opt.gpu>0 then
  network=network:cuda()
end

local data_train = load_datasets(opt.dataset_file, false)
local data_size = #data_train--math.floor(0.9 * #data_train)
local iterations_per_epoch = math.ceil(data_size/opt.batch_size)

local data_test = load_datasets(opt.dataset_test, true)

print('Number of samples ' .. data_size)
print('Number of epochs ' .. math.ceil(opt.max_iter/iterations_per_epoch))
print('Number of iters/epoch ' .. iterations_per_epoch)

hyper_params.learningRateDecay = 1/iterations_per_epoch -- decrease lr by 2 after each epoch
--hyper_params.learningRateDecay = 9/iterations_per_epoch -- decrease lr by 10 after each epoch

function valid(data)
  local pose_error = {}
  for sample = opt.batch_size, #data, opt.batch_size do
    local curr_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local base_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local depth_data = torch.Tensor(opt.batch_size,1,opt.input_height,opt.input_width)
    local pose_data  = torch.Tensor(opt.batch_size,3)
    local camera_data= torch.Tensor(opt.batch_size,4)
    for j=1,opt.batch_size do
      local curr_frame  = image.load(data[sample][4],opt.input_channels,'float')
      local base_frame  = image.load(data[sample][5],opt.input_channels,'float')

      local image_height, image_width = base_frame:size(2), base_frame:size(3)
      local crop_offset  = prepro.center_crop(opt.input_height, opt.input_width, image_height, image_width)
      local camera_model = CameraIntrinsics(image_height, image_width, data[sample][2])

      local depth_frame = image.load(data[sample][3],1,'byte'):float()
      depth_frame  = torch.clamp(depth_frame,0.1,math.huge)
      depth_frame  = torch.pow(depth_frame,-1.0)
      depth_frame  = depth_frame * camera_model:focalLength() * camera_model:baseline()

      camera_data[j] = torch.Tensor(camera_model:intrinsics(crop_offset))
      pose_data[j]   = torch.Tensor(data[sample][1])
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

    network:forward({curr_data, base_data, depth_data, camera_data})
    local pred_pose = model:get_delta_pose_pred()
    pose_error[#pose_error+1] = util.distance(pred_pose,pose_data)
  end
  local mean_error = torch.mean(torch.Tensor(pose_error))
  --local stdv_error = torch.std(torch.Tensor(pose_error))
  return mean_error
end

function train(data)

  network:training()

  -- get weights and loss wrt weights from the model
  params, grad_params = network:getParameters()
  print('Number of parameters ' .. params:nElement())

  local criterion = require('l2_loss')
  if opt.gpu>0 then
    criterion=criterion:cuda()
  end

  local warping_model = build_warping_model(opt.input_height, opt.input_width)
  local pointcloud_model = build_pointcloud_model(opt.input_height, opt.input_width)
  if opt.gpu>0 then
    warping_model=warping_model:cuda()
    pointcloud_model=pointcloud_model:cuda()
  end

  local shuffle = torch.randperm(data_size):long()
  if opt.gpu>0 then
    shuffle = shuffle:cuda()
  end

  function next_batch(iter)
    local curr_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local base_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local depth_data = torch.Tensor(opt.batch_size,1,opt.input_height,opt.input_width)
    local pose_data  = torch.Tensor(opt.batch_size,3)
    local camera_data= torch.Tensor(opt.batch_size,4)
    local samples = {}
    for j=1, opt.batch_size do
      local sample = shuffle[(opt.batch_size*(iter-1)+j-1)%(data_size)+1]

      table.insert(samples, sample)

      local curr_frame  = image.load(data[sample][4],opt.input_channels,'float')
      local base_frame  = image.load(data[sample][5],opt.input_channels,'float')
      ----[[
      curr_frame = prepro.saturation(curr_frame, 0.1)
      base_frame = prepro.saturation(base_frame, 0.1)
      curr_frame = prepro.brightness(curr_frame, 0.1)
      base_frame = prepro.brightness(base_frame, 0.1)
      curr_frame = prepro.contrast(curr_frame, 0.1)
      base_frame = prepro.contrast(base_frame, 0.1)
      --]]
      local image_height, image_width = curr_frame:size(2), curr_frame:size(3)
      local crop_offset  = prepro.center_crop(opt.input_height, opt.input_width, image_height, image_width)
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

    return {pose_data, camera_data, depth_data, curr_data, base_data}, samples
  end

  local smallest_error = math.huge
  local first_loss = nil
  local train_lr = {}
  local train_losses = {}
  local train_errors = {}
  local valid_errors = {}
  for iteration=1,opt.max_iter do
    xlua.progress(iteration, opt.max_iter) 

    if math.fmod(iteration, iterations_per_epoch) == 0 then
      shuffle = torch.randperm(data_size):long() 
      if opt.gpu>0 then
        shuffle = shuffle:cuda()
      end
    end

    local inputs, samples = next_batch(iteration)

    local target = pointcloud_model:forward({inputs[1],inputs[3],inputs[2]}):clone()

    feval = function(x)
      if x ~= params then params:copy(x) end
      grad_params:zero()
      local input = {inputs[4],inputs[5],inputs[3],inputs[2]}
      local output = network:forward(input)
      local loss = criterion:forward(output, target)
      local gradOutputs = criterion:backward(output, target)
      local gradInputs = network:backward(input, gradOutputs)
      return loss, grad_params
    end

    _, loss = optim.adam(feval,params,hyper_params)

    train_lr[iteration] = hyper_params.learningRate / (1 + hyper_params.t * hyper_params.learningRateDecay)

    train_losses[iteration] = loss[1]

    if opt.display>0 and math.fmod(iteration , opt.display) == 0 then
      local true_cloud = target
      local pred_cloud = model:get_warping_output_pred()
      local pred_image = warping_model:forward({inputs[5],pred_cloud,inputs[2]}):clone()
      local true_image = warping_model:forward({inputs[5],true_cloud,inputs[2]}):clone()
      if opt.display==1 then
        for i=1, opt.batch_size do
          pred_window = image.display{image = pred_image[i], zoom = 1, win=pred_window, legend = 'pred'}
          true_window = image.display{image = true_image[i], zoom = 1, win=true_window, legend = 'true'}
        end
      end
    end

    local pred_pose = model:get_delta_pose_pred()
    local true_pose = inputs[1]
    local pose_erro = util.distance(pred_pose, true_pose)
    train_errors[iteration] = pose_erro

    if opt.verbose>0 and iteration % opt.verbose == 0 then
      local grad_params_norm_ratio = grad_params:norm() / params:norm()
      print(string.format("%d, error = %2.4f, loss = %6.8f, grad/param norm = %6.4e", iteration, pose_erro, loss[1], grad_params_norm_ratio))
    end

    if iteration % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.')
      break -- halt
    end

    if first_loss == nil then 
      first_loss = loss[1] 
    end

    if loss[1] > first_loss * 10 then
      print('loss is exploding, aborting.')
      break -- halt
    end

    if loss[1] > 20 then
      fd = io.open(string.format("suspicious_batch_%d.txt", iteration), 'w')
      for sample=1, #samples do
        fd:write(string.format("%s %s %f %f %f\n", data[sample][4], data[sample][5], unpack(data[sample][1])))
      end
      fd:close()
    end

    if opt.save>0 and math.fmod(iteration , opt.save) == 0 then
      network:evaluate()
      valid_errors[#valid_errors+1] = valid(data_test)
      network:training()
      if smallest_error > valid_errors[#valid_errors] then
        smallest_error = valid_errors[#valid_errors]
        --torch.save(string.format("best_model_%d.t7", iteration), network)
        torch.save("best_model.t7", network)
      end
    end

    if opt.save>0 and math.fmod(iteration , opt.save) == 0 then
      torch.save("train_lr.t7", train_lr)
      torch.save("train_loss.t7", train_losses)
      torch.save("train_error.t7", train_errors)
      torch.save("valid_error.t7", valid_errors)
      torch.save("train_state.t7", hyper_params)
      torch.save("train_model.t7", network)
    end

    --[[
    local fd = io.open('./dump.txt', 'a+')
    local grad_pose = model:get_delta_pose_grad()
    for b=1,opt.batch_size do 
      for i=1,3 do 
        fd:write(string.format("%.4f ", true_pose[b][i]-pred_pose[b][i])) 
      end 
      for i=1,3 do 
        fd:write(string.format("%.4f ", grad_pose[b][i]))
      end 
      fd:write("\n")
    end
    fd:close()
    --]]
  end
  print ('Training done')
end

train(data_train)
