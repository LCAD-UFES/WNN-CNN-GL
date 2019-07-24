#! /usr/bin/env luajit

require 'torch'

function delta_datasetname(dir, year_base, year_live, offset_base, offset_live)
  return string.format("%s/deltapos-%s-%s-%sm-%sm.txt", dir, year_base, year_live, offset_base, offset_live)
end

learning_rate = {1e-5, 1e-4, 1e-6, 1e-4, 1e-4, 1e-4}--, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4}
offset_base = {'0', '1', '2', '3', '4', '5'}--, '6.0', '7.0', '8.0', '9.0', '10.0'}
offset_curr = '0'
output_dir = '/home/avelino/deepslam/data/ufes_cnn'
datasets = {'20160825', '20160825-01', '20160825-02', '20161021', '20171122', '20171205'}
--os.execute("rm dump.txt")

local errors = {}
for k=3, #offset_base do
  local dataset_name = delta_datasetname(output_dir, '20160830', '20160830', offset_base[k], '0')
  
  local dataset_test = io.open('dataset_test.txt', 'w')
  dataset_test:write(dataset_name..'\n')
  dataset_test:close()
  
  local dataset_train = io.open('dataset_list.txt', 'w')
  for i=1, #datasets do  -- base datasets
    for j=1, #datasets do  -- live datasets
      if (offset_base[k] ~= offset_curr) or ((offset_base[k] == offset_curr) and (i == j)) then 
        dataset_name = delta_datasetname(output_dir, datasets[i], datasets[j], offset_base[k], offset_curr)
        dataset_train:write(dataset_name)
        if i*j < (#datasets * #datasets) then 
          dataset_train:write('\n') 
        end
        print(dataset_name)
      end
    end
  end
  dataset_train:close()

  local lr = learning_rate[k]
  local iters = 2e3
  local resume = ' -resume train_model.t7'
  if k == 1 then iters = 5e3; resume = '' end
  if k == #offset_base then iters = 25e3 end

  os.execute("th run_train.lua -save 200 -batch_size 80 -learning_rate "..tostring(lr).." -max_iter "..tostring(iters)..resume)
  os.execute("th plot_loss.lua && mv plot_loss.png plot_loss"..tostring(k-1)..".png")
  --os.execute("th convert_model.lua")
  --local test_error = dofile('run_inference.lua')
  --errors[#errors+1] = test_error
  --print(k, test_error)
end

--torch.save("curriculum_error.t7", errors, 'ascii')
