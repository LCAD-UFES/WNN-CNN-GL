require 'nn'
require 'gvnn'
require 'TransformationMatrix3x4GeneratorSE2'
require 'Transform3DPoints_RtK'

-- TODO refactor Transform3DPoints_Rt to accept camera params as input instead of on the constructor
function build_pointcloud_model(height, width)
  -- converts the 6-vector (3-vector so3 for rotation and 3-vector for translation)
  local Rt_net = nn.Sequential()
  Rt_net:add(nn.SelectTable(1)) --6D/3D input
  --Rt_net:add(nn.TransformationMatrix3x4SO3(true,false,true))
  Rt_net:add(nn.TransformationMatrix3x4SE2(true,false,true))

  local depth = nn.Sequential()
  depth:add(nn.SelectTable(2)) --depth input

  local concat_Rt_depth = nn.ConcatTable()
  concat_Rt_depth:add(Rt_net)
  concat_Rt_depth:add(depth)
  concat_Rt_depth:add(nn.SelectTable(3)) --camera params

  local Transformation3x4net = nn.Sequential()
  Transformation3x4net:add(concat_Rt_depth)
  Transformation3x4net:add(nn.Transform3DPoints_RtK(height, width))
  --Transformation3x4net:add(nn.Transform3DPoints_Rt(height, width, 707.0912, 707.0912, 601.8873, 183.1104))
  return Transformation3x4net --return 3D points for computing L2 loss in meters
end