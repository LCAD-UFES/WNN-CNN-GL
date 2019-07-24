local TransformationMatrix3x4SE2, parent = torch.class('nn.TransformationMatrix3x4SE2', 'nn.TransformationMatrix3x4SO3')

function TransformationMatrix3x4SE2:__init(useRotation, useScale, useTranslation)
  parent.__init(self, useRotation, useScale, useTranslation)
end

function TransformationMatrix3x4SE2:check(input)
  --we use an over parametrised version of SE(2) with rx=0, rz=0 and ty=0
  parent.check(self, input)
end

function TransformationMatrix3x4SE2:updateOutput(_transformParams)
  local batchSize = _transformParams:size(1)
  local transformParams = torch.zeros(batchSize, 6):typeAs(_transformParams)
  transformParams:select(2,2):copy( _transformParams:select(2,1) ) --ry
  transformParams:select(2,4):copy( _transformParams:select(2,2) ) --tx
  transformParams:select(2,6):copy( _transformParams:select(2,3) ) --tz
  parent.updateOutput(self, transformParams)
  return self.output
end

function TransformationMatrix3x4SE2:updateGradInput(_transformParams, _gradParams)
  local batchSize = _transformParams:size(1)
  local transformParams = torch.zeros(batchSize, 6):typeAs(_transformParams)
  transformParams:select(2,2):copy( _transformParams:select(2,1) ) --ry
  transformParams:select(2,4):copy( _transformParams:select(2,2) ) --tx
  transformParams:select(2,6):copy( _transformParams:select(2,3) ) --tz
  parent.updateGradInput(self, transformParams, _gradParams)
  local _gradInput = self.gradInput:clone()
  self.gradInput:resize(batchSize, 3)
  self.gradInput:select(2,1):copy( _gradInput:select(2,2) ) --ry
  self.gradInput:select(2,2):copy( _gradInput:select(2,4) ) --tx
  self.gradInput:select(2,3):copy( _gradInput:select(2,6) ) --tz
  return self.gradInput
end
