require 'torch'
require 'gnuplot'
local csv = require("csv")

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:text("Options")
cmd:option('-batch_size', 10, 'Batch size')

local opt = cmd:parse(arg)
--[[
local f = csv.open('./dump.txt', {separator=' '})
for fields in f:lines() do
  for i, value in ipairs(fields) do
    if i == 1 then erro_ry = tonumber(value) end 
    if i == 2 then erro_tx = tonumber(value) end 
    if i == 3 then erro_tz = tonumber(value) end 
    if i == 4 then grad_ry = tonumber(value) end 
    if i == 5 then grad_tx = tonumber(value) end 
    if i == 6 then grad_tz = tonumber(value) end 
  end
end
--]]
gnuplot.pngfigure('plot_grad.png')
gnuplot.title('Plot of errors/grads w.r.t Pose')
--gnuplot.raw("plot 'dump.txt' using 0:1 title 'erro:ry' with lines, 'dump.txt' using 0:4 title 'grad:ry' with lines")
--gnuplot.raw("plot 'dump.txt' using 0:2 title 'erro:tx' with lines, 'dump.txt' using 0:5 title 'grad:tx' with lines")
gnuplot.raw("plot 'dump.txt' using 0:3 title 'erro:tz' with lines, 'dump.txt' using 0:6 title 'grad:tz' with lines")
gnuplot.xlabel('iterations')
gnuplot.ylabel('error')
gnuplot.grid(true)
gnuplot.plotflush()