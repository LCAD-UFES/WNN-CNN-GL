require 'torch'
local grad = require 'autograd'
grad.optimize(true) -- global

local loss_func = function(output, target)
  local coord_pred = output:view(-1,3)
  local coord_true = target:view(-1,3)
  local coord_diff = coord_pred - coord_true
  local coord_abs = torch.abs(coord_diff)
  local coord_sum = torch.sum(coord_abs,2)
  local MAE = torch.mean(coord_sum) --mean distance over mini-batches
  local loss = MAE --+ torch.normal(0.0,0.2)
  return loss, coord_pred
end

return grad.nn.AutoCriterion('AutoMAE')(loss_func)