function [direc, shape] = parse_inputs(varargin)
direc = 'lin';
shape = 'same';
flag = [0 0]; % [dir shape]

for i = 1 : nargin
   t = varargin{i};
   if strcmp(t,'col') && flag(1) == 0
      direc = 'col';
      flag(1) = 1;
   elseif strcmp(t,'full') && flag(2) == 0
      shape = 'full';
      flag(2) = 1;
   elseif strcmp(t,'same') && flag(2) == 0
      shape = 'same';
      flag(2) = 1;
   elseif strcmp(t,'valid') && flag(2) == 0
      shape = 'valid';
      flag(2) = 1;
   else
      error(['Too many / Unkown parameter : ' t ])
   end
end