require 'torch'
require 'gnuplot'

local losses = torch.load('train_loss.t7')
local errors = torch.load('train_error.t7')
local valid = torch.load('valid_error.t7')
losses = torch.Tensor(losses)
errors = torch.Tensor(errors)
table.insert(valid, 1, errors[1]) --initial validation assumed to be equal as training error
valid_errors = torch.Tensor(valid)
local iters = errors:size(1)
gnuplot.figure(1)
--gnuplot.pngfigure('plot_loss.png')
gnuplot.title('Loss/error minimisation over time')
gnuplot.plot({'Loss',torch.linspace(1,iters,iters),losses,'~'},{'Train',torch.linspace(1,iters,iters),errors,'~'},{'Valid',torch.linspace(1,iters,#valid),valid_errors,'~'})
--gnuplot.plotflush()
