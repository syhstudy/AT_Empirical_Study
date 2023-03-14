-- Adapted from https://github.com/wojciechz/learning_to_execute

--[[
--
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--
local MaskedLoss, parent = torch.class('MaskedLoss', 'nn.Module')

function MaskedLoss:__init()
   parent.__init(self)
end

function MaskedLoss:updateOutput(input)
  local input, target = unpack(input)
  local output = 0
  local acc = 0
  local normal = 0
  for i = 1, target:size(1) do
    if target[i] ~= 1 then
      if input[i]:max() == input[i][target[i]] then
        acc = acc + 1
      end
      normal = normal + 1
      output = output - input[i][target[i]]
    end
  end
  output = output / params.batch_size  -- target:size(1)
  self.output = {output, acc, normal}
  return self.output
end

function MaskedLoss:updateGradInput(input)
	self.gradInput = {[1] = torch.Tensor():cuda(), [2] = torch.Tensor():cuda()}
  local input, target = unpack(input)
  self.gradInput[1]:resizeAs(input)
  self.gradInput[1]:zero()
  self.gradInput[2]:resizeAs(target)
  self.gradInput[2]:zero()

  local z = -1 / params.batch_size  -- target:size(1)
  local gradInput = self.gradInput[1]
  for i=1,target:size(1) do
    if target[i] ~= 1 then
      gradInput[i][target[i]] = z
    end
  end
  return self.gradInput
end
