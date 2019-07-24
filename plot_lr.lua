require 'torch'
require 'gnuplot'

local lr = torch.load('train_lr.t7')
lr = torch.Tensor(lr)
local iters = lr:size(1)
gnuplot.figure(1)
--gnuplot.pngfigure('plot_lr.png')
gnuplot.title('Learning rate regime over time')
gnuplot.plot({'Learning rate',torch.linspace(1,iters,iters),lr,'~'})
--gnuplot.plotflush()
