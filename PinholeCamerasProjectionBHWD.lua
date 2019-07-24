local PinholeCamerasProjectionBHWD, parent = torch.class('nn.PinholeCamerasProjectionBHWD', 'nn.Module')

--[[

   PinholeCamerasProjectionBHWD(height, width, fx, fy, u0, v0) :
   PinholeCamerasProjectionBHWD:updateOutput(points3D)
   PinholeCamerasProjectionBHWD:updateGradInput(points3D, gradGrids)

   PinholeCamerasProjectionBHWD will take b x h x w x 3 3D points and returns 
   a projection of that on a 2D plabe as b x h x w x 2
	
   for any 3D points repesented as (X, Y, Z) the projection is 

   fx * ( X / Z ) + u0
   fy * ( Y / Z ) + v0 

   where fx and fy are focal lengths and (u0, v0) is camera center

]]

function PinholeCamerasProjectionBHWD:__init(height, width)
  parent.__init(self)

  assert(height > 1)
  assert(width > 1)

  self.height = height
  self.width  = width

  self.epsilon = 1e-12

  self.gradInput = {}  
  self.gradInput[1] = torch.Tensor(1)
  self.gradInput[2] = torch.Tensor(1)

  self.output = torch.Tensor(1)
end

function PinholeCamerasProjectionBHWD:updateCamera(intrinsics)
  local fx = intrinsics[{{},1}]
  local fy = intrinsics[{{},2}]
  local u0 = intrinsics[{{},3}]
  local v0 = intrinsics[{{},4}]

  self.u0 = -1 + 2 * (u0-1)/(self.width-1) 
  self.v0 = -1 + 2 * (v0-1)/(self.height-1)

  self.fx =  2 * fx/(self.width-1)
  self.fy =  2 * fy/(self.height-1)

  --[[ what we want is this 
     	u' = -1 + ( fx*(X/Z) + u0 - 1) / (w - 1 ) * 2 
     	u' = -1 + 2 * (u0 - 1) / (w - 1) + 2*fx/(w-1) * (X/Z) 
     ]]--
end

function PinholeCamerasProjectionBHWD:updateOutput(points3D_intrinsics)
  local points3D

  _points3D, _intrinsics = unpack(points3D_intrinsics)

  self:updateCamera(_intrinsics)

  points3D = _points3D

  assert(points3D:nDimension()==4
    and points3D:size(2)==self.height
    and points3D:size(3)==self.width
    and points3D:size(4)==3
    , 'please input 3d points in the format (bxhxwx3)')
  local batchsize = points3D:size(1)

  self.output:resize(points3D:size(1), points3D:size(2), points3D:size(3), 2):zero()
  --self.output:resizeAs(points3D):zero()

  local X = points3D:select(4,1) 
  local Y = points3D:select(4,2) 
  local Z = points3D:select(4,3) + self.epsilon

  --[[
    u'  = fx * ( X / Z ) + u0
    v'  = fy * ( Y / Z ) + v0
  --]]
  for b=1,batchsize do
    self.output[b]:select(3,1):copy(torch.mul(torch.cdiv(X,Z)[b],self.fx[b]) + self.u0[b])
    self.output[b]:select(3,2):copy(torch.mul(torch.cdiv(Y,Z)[b],self.fy[b]) + self.v0[b])
  end

  if _points3D:nDimension()==2 then
    self.output = self.output:select(1,1)
  end

  return self.output
end

--[[
	Pi(X, Y, Z ) = (X/Z, Y/Z) 

        Pi projects the 3D dimensional point to 2D plane
 
	dE    dE   dPi   [dE1 dE2] x [ 1/Z 0 ]
        --  = -- x --  = 
        dX    dPi  dX 

	dE    dE   dPi   [dE1 dE2] x [ 0 1/Z ]
        --  = -- x --  = 
        dY    dPi  dY 
    	
	dE    dE   dPi   [dE1 dE2] x [-X/Z^2  -Y/Z^2 ]
        --  = -- x --  = 
        dZ    dPi  dZ 

--]]

function PinholeCamerasProjectionBHWD:updateGradInput(_input, _gradOut)
  local _points3D = _input[1]
  local intrinsics = _input[2]
  local points3D, gradGrid
  --TODO check dimensions of intrinsics operations
  if _points3D:nDimension()==2 then
    points3D = addOuterDim(_points3D)
    gradGrid = addOuterDim(_gradOut)
  else
    points3D = _points3D
    gradOut = _gradOut
  end

  local batchsize = points3D:size(1)

  self.gradInput[1]:resizeAs(points3D):zero():expandAs(points3D)

  self.gradInput[2]:resizeAs(intrinsics):zero():expandAs(intrinsics)

  local X = points3D:select(4,1)
  local Y = points3D:select(4,2)
  local Z = points3D:select(4,3)
  local Zs = points3D:select(4,3)

  local X_div_Z = torch.cdiv(-X,Z)
  local Y_div_Z = torch.cdiv(-Y,Z)

  Zs = Zs:add(self.epsilon)

  local dLx = gradOut:select(4,1) 
  local dLy = gradOut:select(4,2) 

  local dLx_div_Z = torch.mul(torch.cdiv(dLx,Zs),self.fx)  
  local dLy_div_Z = torch.mul(torch.cdiv(dLy,Zs),self.fy)  

  self.gradInput[1]:select(4,1):copy(dLx_div_Z)
  self.gradInput[1]:select(4,2):copy(dLy_div_Z)
  self.gradInput[1]:select(4,3):copy(torch.cmul(dLx_div_Z,X_div_Z) + torch.cmul(dLy_div_Z,Y_div_Z))

  if _points3D:nDimension()==2 then
    self.gradInput[1] = self.gradInput[1]:select(1,1)
  end

  return self.gradInput
end
