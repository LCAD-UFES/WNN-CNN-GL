require 'torch'
require 'gvnn'

local prepro = {}

local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
local hybrid_mean = torch.FloatTensor{0.554, 0.486, 0.439}:view(1,3,1,1) -- in RGB order
local hybrid_std  = torch.FloatTensor{0.314, 0.314, 0.314}:view(1,3,1,1) -- in RGB order

-- takes a batch of images and preprocesses them
function prepro.crop_image(img, offset, height, width)
  local output = torch.Tensor(img:size(1), height, width):typeAs(img)
  local h, w = img:size(2), img:size(3)
  local xoff = offset[1]
  local yoff = offset[2]
  local crop = img:narrow(2,yoff,height):narrow(3,xoff,width)
  output:copy(crop)
  assert(output:size(2) == height)
  assert(output:size(3) == width)
  return output
end

function prepro.center_crop(input_height, input_width, image_height, image_width)
  local offset = {}
  local xoff, yoff
  -- sample center
  -- xoff, yoff = math.ceil((image_width-input_width)/2), math.ceil((h-height)/2)
  -- sample bottom center
  xoff, yoff = math.ceil((image_width-input_width)/2), (image_height-input_height)
  offset[1] = xoff +1
  offset[2] = yoff +1
  return offset
end

function prepro.random_crop(input_height, input_width, image_height, image_width)
  local offset = {}
  local xoff, yoff
  -- sample randomly
  --xoff = torch.random( 1, image_width-input_width+1 ) 
  xoff = torch.random( 0.49*(image_width-input_width+1), 0.51*(image_width-input_width+1) ) 
  yoff = torch.random( 0.95*(image_height-input_height+1), 1.0*(image_height-input_height+1) )
  offset[1] = xoff -- random returns values between [1,dx+1]
  offset[2] = yoff -- random returns values between [1,dy+1]
  return offset
end

function prepro.normalize_hybrid(imgs)
  hybrid_mean = hybrid_mean:typeAs(imgs) 
  hybrid_std = hybrid_std:typeAs(imgs) 
  imgs:add(-hybrid_mean:expandAs(imgs))
  imgs:cdiv(hybrid_std:expandAs(imgs))
  return imgs
end

function prepro.denormalize_hybrid(imgs)
  hybrid_mean = hybrid_mean:typeAs(imgs) 
  hybrid_std = hybrid_std:typeAs(imgs) 
  imgs:cmul(hybrid_std:expandAs(imgs))
  imgs:add(hybrid_mean:expandAs(imgs))
  return imgs
end

function prepro.denormalize(imgs)
  return imgs:mul(255.0):add(128.0)
end

function prepro.normalize(imgs)
  return imgs:add(-128.0):div(255.0)
end

local function blend(img1, img2, alpha)
  return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
  dst:resizeAs(img)
  dst[1]:zero()
  dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
  dst[2]:copy(dst[1])
  dst[3]:copy(dst[1])
  return dst
end

function prepro.saturation(input, var)
  local gs

  gs = gs or input.new()
  grayscale(gs, input)

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

function prepro.brightness(input, var)
  local gs

  gs = gs or input.new()
  gs:resizeAs(input):zero()

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

function prepro.contrast(input, var)
  local gs

  gs = gs or input.new()
  grayscale(gs, input)
  gs:fill(gs[1]:mean())

  local alpha = 1.0 + torch.uniform(-var, var)
  blend(input, gs, alpha)
  return input
end

--[[
credits: https://groups.google.com/forum/#!topic/torch7/fgQ9vhKKUNE
Below is a function that I wrote using the stnbhwd library that upsamples batches of images on the GPU.  The way it works is a bit non-intuitive because we are not using the library for its primary purpose, but rather just taking advantage of its bilinear interpolator.  The way the function works is that it fixes the grid generator parameters to perform the identity operation, because we're not trying to perform rotations or translations, and we are not interested in scaling up some subsample of the image.  Rather we want to scale the whole image, and that information is provided as arguments to  nn.AffineGridGeneratorBHWD, which is the module that creates the grid for the spatial sampler.  I named the function upsample, but I expect it should downsample just as well.
--]]
-- scale up/down batches of images on GPU
function prepro.scale(maps_BDHW, upsample_height, upsample_width)
    
    -- check if the maps are batches (i.e., 4 dimesional)
    local maps_dim = maps_BDHW:nDimension()
    if maps_dim == 3 then
        maps_BDHW = maps_BDHW:view(1, maps_BDHW:size(1), maps_BDHW:size(2), maps_BDHW:size(3))
    end
    
    -- convert BDHW to BHWD convention
    maps_BHWD = nn.Transpose({2,3},{3,4}):forward(maps_BDHW)
    
    -- define grid generator parameters (to perform the identity transformation)
    local batch_size = maps_BHWD:size(1)
    grid_generator_params = torch.Tensor(batch_size, 2, 3):fill(0)
    for i = 1, batch_size do grid_generator_params[i][1][1] = 1; grid_generator_params[i][2][2] = 1 end
    
    -- creat the sampling grid for identity transformation with output size upsample_height x upsample_width
    local sampling_grid = nn.AffineGridGeneratorBHWD(upsample_height, upsample_width):forward(grid_generator_params)
    scaled_maps_BHWD = nn.BilinearSamplerBHWD():forward({maps_BHWD, sampling_grid})
    
    -- convert back to BDHW convention
    scaled_maps_BDHW = nn.Transpose({3,4},{2,3}):forward(scaled_maps_BHWD)
    
    -- if input was a single unbatched map, convert back to 3 dimensions 
    if maps_dim == 3 then
        scaled_maps_BDHW = scaled_maps_BDHW:view(scaled_maps_BDHW:size(2), scaled_maps_BDHW:size(3), scaled_maps_BDHW:size(4))
    end
    
    -- return scaled feature maps
    return scaled_maps_BDHW
    
end
return prepro