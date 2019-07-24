-- Utilities
local util = {}

local cast
function cast(tableOfParams, typeName)
   -- Some nice aliases
   if typeName == "float" then typeName = "torch.FloatTensor" end
   if typeName == "double" then typeName = "torch.DoubleTensor" end
   if typeName == "cuda" then typeName = "torch.CudaTensor" end
   if typeName == "cudaDouble" then typeName = "torch.CudaDoubleTensor" end

   -- If we passed in a tensor, just cast it
   if torch.isTensor(tableOfParams) then
      return tableOfParams:type(typeName)
   end

   -- Recursively cast
   local out = {}
   for key,value in pairs(tableOfParams) do
      if torch.isTensor(value) then
         out[key] = value:type(typeName)
      elseif type(value) == "table" then
         out[key] = cast(value,typeName)
      else
         out[key] = value
      end
   end
   return out
end
util.cast = cast

local distance 
function distance(output, target)
  assert(output:dim() <= 2, 'up to 2d tensor only')
  if output:dim() == 1 then
    output = output:view(1,-1)
    target = target:view(1,-1)
  end
  --[[
  local dx = output[{{}, {4}}] - target[{{}, {4}}]
  local dy = output[{{}, {5}}] - target[{{}, {5}}]
  local dz = output[{{}, {6}}] - target[{{}, {6}}]
  local position_diff = torch.cat({dx, dy, dz}, 2)
  --]]
  ----[[
  local dx = output[{{}, {2}}] - target[{{}, {2}}]
  local dz = output[{{}, {3}}] - target[{{}, {3}}]
  local position_diff = torch.cat({dx, dz}, 2)
  --]]

  local euclidean_dist = torch.sqrt(torch.sum(torch.cmul(position_diff, position_diff),2))

  return torch.mean(euclidean_dist) --mean distance over mini-batches
end
util.distance = distance

return util