require 'torch'
local grad = require 'autograd'
grad.optimize(true) -- global

local epsilon = 1e-10
local loss_func = function(output, target)
  local coord_pred = output:view(-1,3)
  local coord_true = target:view(-1,3)
  local coord_diff = coord_pred - coord_true
  local coord_square = torch.cmul(coord_diff, coord_diff)
  local coord_sum = torch.sum(coord_square,2)
  local euclidean_distance = torch.sqrt(coord_sum+epsilon)
  local mean_euclidean_distance = torch.mean(euclidean_distance) --mean distance over mini-batches
  --local stdv_euclidean_distance = torch.sqrt(torch.mean(torch.pow(euclidean_distance - mean_euclidean_distance, 2))+epsilon)
  local loss = mean_euclidean_distance --stdv_euclidean_distance --torch.normal(0.01,0.20)
  return loss, coord_pred
end

return grad.nn.AutoCriterion('AutoEuclideanDistance')(loss_func)