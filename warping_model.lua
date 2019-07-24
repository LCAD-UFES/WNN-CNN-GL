require 'nn'
require 'gvnn'
require 'PinholeCamerasProjectionBHWD'

-- TODO refactor PinHoleCameraProjectionBHWD to accept camera params as input instead of on the constructor
function build_warping_model(height, width)
  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet=nn.Sequential()
  tranet:add(nn.SelectTable(1)) --RGB input
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2,3},{3,4}))

  local projection_input = nn.ConcatTable()
  projection_input:add(nn.SelectTable(2)) --Point Cloud input
  projection_input:add(nn.SelectTable(3)) --Camera Intrinsics
  
  local Transformation3x4net = nn.Sequential()
  Transformation3x4net:add(projection_input)
  Transformation3x4net:add(nn.PinholeCamerasProjectionBHWD(height, width))
  --Transformation3x4net:add(nn.SelectTable(2)) --Point Cloud input
  --Transformation3x4net:add(nn.PinHoleCameraProjectionBHWD(height, width, 707.0912, 707.0912, 601.8873, 183.1104))
  Transformation3x4net:add(nn.ReverseXYOrder())

  local concat = nn.ConcatTable()
  concat:add(tranet)
  concat:add(Transformation3x4net)

  local warping_net = nn.Sequential()
  warping_net:add(concat)
  warping_net:add(nn.BilinearSamplerBHWD())
  warping_net:add(nn.Transpose({3,4},{2,3}))

  return warping_net
end