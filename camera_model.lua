require 'torch'

do
  local CameraIntrinsics = torch.class('CameraIntrinsics')

  function CameraIntrinsics:__init(height, width, params)
    self.fx, self.cx, self.fy, self.cy, self.b = unpack(params)
    self.fx = self.fx * width
    self.cx = self.cx * width
    self.fy = self.fy * height
    self.cy = self.cy * height
  end

  function CameraIntrinsics:intrinsics(offset, scale)
    scale = scale or 1
    local params = {}
    params[1] = self.fx / scale
    params[2] = self.fy / scale
    params[3] = (self.cx - offset[1] + 1) / scale --xoff
    params[4] = (self.cy - offset[2] + 1) / scale --yoff
    return params
  end

  function CameraIntrinsics:focalLength()
    return self.fx
  end

  function CameraIntrinsics:baseline()
    return self.b
  end

end
