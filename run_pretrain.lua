require 'optim' 
require 'image'
require 'dataset_load'
require 'siamese_model3'
local prepro = require 'prepro'
local util = require 'util'

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:text("Options")
cmd:option("-dataset_file", "dataset_train.txt", "dataset file")
cmd:option("-margin", 100, "criterion margin")
cmd:option("-input_width", 320, "input width")
cmd:option("-input_height", 240, "input height")
cmd:option("-input_channels", 3, "input channels")
cmd:option('-gpu', 1, 'GPU to use. 0 = no GPU')
cmd:option('-batch_size',100, 'Batch size')
cmd:option('-max_iter',1e4,'number of iterations')
cmd:option('-learning_rate', 1e-2,'learning rate')
cmd:option('-verbose', 1,'Print messages interval')
cmd:option('-display', 0,'Display images interval')
cmd:option('-save', 200,'Save training interval')
cmd:option('-resume','','Resume from a model')

local opt = cmd:parse(arg)

local hyper_params = {
  learningRate = opt.learning_rate,
  learningRateDecay = 0, --set below
  weightDecay = 0,
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

local network = build_network_model_pretrain(opt.gpu, true)

if opt.gpu>0 then
  network=network:cuda()
end

local data_train = load_pretrain_datasets(opt.dataset_file)
local data_size = #data_train
local iterations_per_epoch = math.ceil(data_size/opt.batch_size)

print('Number of samples ' .. data_size)
print('Number of epochs ' .. math.ceil(opt.max_iter/iterations_per_epoch))
print('Number of iters/epoch ' .. iterations_per_epoch)

hyper_params.learningRateDecay = 9/iterations_per_epoch -- decrease lr by 10 after each epoch

function pretrain(data)

  network:training()

  -- get weights and loss wrt weights from the model
  params, grad_params = network:getParameters()
  print('Number of parameters ' .. params:nElement())

  local criterion = nn.HingeEmbeddingCriterion(opt.margin)
  if opt.gpu>0 then
    criterion=criterion:cuda()
  end

  local shuffle = torch.randperm(data_size):long()
  if opt.gpu>0 then
    shuffle = shuffle:cuda()
  end
  
  function next_batch(iter)
    local curr_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local base_data  = torch.Tensor(opt.batch_size,opt.input_channels,opt.input_height,opt.input_width)
    local target_data = torch.Tensor(opt.batch_size,1)

    for j=1, opt.batch_size do
      local sample = shuffle[(opt.batch_size*(iter-1)+j-1)%(data_size)+1]
      local curr_frame  = image.load(data[sample][2],opt.input_channels,'float')
      local base_frame  = image.load(data[sample][3],opt.input_channels,'float')
      
      curr_frame = prepro.saturation(curr_frame, 0.1)
      base_frame = prepro.saturation(base_frame, 0.1)
      curr_frame = prepro.brightness(curr_frame, 0.1)
      base_frame = prepro.brightness(base_frame, 0.1)
      curr_frame = prepro.contrast(curr_frame, 0.1)
      base_frame = prepro.contrast(base_frame, 0.1)
      
      local image_height, image_width = curr_frame:size(2), curr_frame:size(3)
      local crop_offset  = prepro.center_crop(opt.input_height, opt.input_width, image_height, image_width)
      target_data[j] = data[sample][1]
      curr_data[j]   = prepro.crop_image(curr_frame, crop_offset, opt.input_height, opt.input_width)
      base_data[j]   = prepro.crop_image(base_frame, crop_offset, opt.input_height, opt.input_width)
    end

    if opt.gpu>0 then
      target_data = target_data:cuda()
      curr_data   = curr_data:cuda()
      base_data   = base_data:cuda()
    end
    
    return {target_data, curr_data, base_data}
  end

  local first_loss = nil
  local train_lr = {}
  local train_losses = {}
  for iteration=1,opt.max_iter do
    xlua.progress(iteration, opt.max_iter) 

    if math.fmod(iteration, iterations_per_epoch) == 0 then
      shuffle = torch.randperm(data_size):long() 
      if opt.gpu>0 then
        shuffle = shuffle:cuda()
      end
    end

    local inputs = next_batch(iteration)

    local target = inputs[1]

    feval = function(x)
      grad_params:zero()
      local input = {inputs[2],inputs[3]}
      local output = network:forward(input)
      local loss = criterion:forward(output, target)
      local gradOutputs = criterion:backward(output, target)
      local gradInputs = network:backward(input, gradOutputs)
      return loss, grad_params
    end

    _, loss = optim.adam(feval,params,hyper_params)

    train_lr[iteration] = hyper_params.learningRate / (1 + hyper_params.t * hyper_params.learningRateDecay)

    train_losses[iteration] = loss[1]

    if opt.verbose>0 and iteration % opt.verbose == 0 then
      local grad_params_norm_ratio = grad_params:norm() / params:norm()
      print(string.format("%d, loss = %6.8f, grad/param norm = %6.4e", iteration, loss[1], grad_params_norm_ratio))
    end

    if iteration % 10 == 0 then collectgarbage() end
   
    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.')
      break -- halt
    end

    if first_loss == nil then first_loss = loss[1] end
    if loss[1] > first_loss * 10 then
      print('loss is exploding, aborting.')
      break -- halt
    end
    
    if opt.save>0 and math.fmod(iteration , opt.save) == 0 then
      torch.save("pretrain_lr.t7", train_lr)
      torch.save("pretrain_loss.t7", train_losses)
      torch.save("pretrain_state.t7", hyper_params)
      torch.save("pretrain_model.t7", network)
    end

  end
  print ('Pretraining done')
end

pretrain(data_train)
