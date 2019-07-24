local csv = require("csv")
local path = require 'pl.path'

function table.shuffle(t)
  for n = #t, 1, -1 do
    local k = math.random(n)
    t[n], t[k] = t[k], t[n]
  end
  return t
end

function table.merge(t1, t2)
  for k,v in ipairs(t2) do
    table.insert(t1, v)
  end  
  return t1
end

function load_dataset(csv_name, inference)
  inference = inference or false
  local data = {}
  local x, y, z, rx, ry, rz
  local base_depth, base_image_left, base_image_right
  local curr_depth, curr_image_left, curr_image_right
  local f = csv.open(csv_name, {separator=' '})
  local j = -1
  if f ~= nil then
    for fields in f:lines() do
      if j ~= -1 then --skip header
        local base_params = {}
        local curr_params = {}
        for i, value in ipairs(fields) do
          if i == 1 then x = tonumber(value) end 
          if i == 2 then y = tonumber(value) end 
          if i == 3 then z = tonumber(value) end 
          if i == 4 then rx = tonumber(value) end 
          if i == 5 then ry = tonumber(value) end 
          if i == 6 then rz = tonumber(value) end 
          if i == 19 then base_depth = value end 
          if i == 20 then base_image_left = value end 
          if i == 21 then base_image_right = value end 
          if i == 22 then curr_depth = value end 
          if i == 23 then curr_image_left = value end 
          if i == 24 then curr_image_right = value end 
          if i >= 25 and i <= 29 then base_params[#base_params+1] = tonumber(value) end 
          if i >= 30 and i <= 34 then curr_params[#curr_params+1] = tonumber(value) end 
        end
        if (inference or path.exists(curr_depth)) and path.exists(base_image_left) and path.exists(curr_image_left) then
          --local delta_pose = {rx, ry, rz, x, y, z}
          local delta_pose = {ry, x, z}
          table.insert(data, {delta_pose, curr_params, curr_depth, curr_image_left, base_image_left})
          if inference == false then
            table.insert(data, {delta_pose, curr_params, curr_depth, curr_image_right, base_image_right})
          end
        end
      end
      j = j + 1
    end
  else
    print('File not found: '..csv_name)
  end
  return data
end

function load_datasets(txt_name, inference)
  inference = inference or false
  local data = {}
  for line in io.lines(txt_name) do
    local temp = load_dataset(line, inference)
    data = table.merge(data, temp)
  end
  if not inference then
    data = table.shuffle(data)
  end
  return data
end

function load_pretrain_datasets(txt_name)
  local data = {}
  for line in io.lines(txt_name) do
    local temp = load_dataset(line, inference)
    local size = #temp
    for positive = 1, size do
      local negative = math.random(size)
      if math.abs(positive - negative) < 100 then
        negative = ((negative + 100) % size) + 1
      end
      table.insert(data, { 1, temp[positive][4], temp[positive][5]})
      --table.insert(data, {-1, temp[negative][4], temp[positive][5]})
      table.insert(data, {-1, temp[positive][4], temp[negative][5]})
    end
  end
  data = table.shuffle(data)
  return data
end
