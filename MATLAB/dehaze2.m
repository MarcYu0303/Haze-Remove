clc; clear;close all; 
%%
% 直方图均衡
% I = imread('C:\Users\yijing sun\Desktop\P1\db\IEI2019\H26.jpg');       % 读取图像
% R = I(:,:,1);  G = I(:,:,2);  B = I(:,:,3);   % 提取红绿蓝三个通道
% %  对三个通道分别进行均衡化处理后整合为三维图像
% M = histeq(R,250);  N = histeq(G,250);  L = histeq(B,250);   In = cat(3, M, N, L);
% figure;
%  subplot(2, 2, 1); imshow(I); title('原图像', 'FontWeight', 'Bold');
%  subplot(2, 2, 2); imshow(In); title('处理后的图像', 'FontWeight', 'Bold');
%     % 灰度化处理
%     Q = rgb2gray(I);      W = rgb2gray(In);
%     % 对图形进行均衡化处理，250为输出图像的灰度级数 
%     subplot(2, 2, 3);  imhist(Q, 250); title('原灰度直方图', 'FontWeight', 'Bold');
%     subplot(2, 2, 4);  imhist(W, 250); title('处理后的灰度直方图', 'FontWeight', 'Bold');

%%
%dark channel prior
Image=imread( 'D:\360MoveData\Users\MarcYu\Desktop\DIP project\db\IEI2019\H18.jpg');                % 读取图像
% imshow(Image), title('原始图像');           % 展示原图
[height ,width, c]=size(Image);             % size求取矩阵大小，返回其行列值，即hight、width
dark_I = zeros(height,width);               % zeros返回 hight x width 大小的零矩阵
for y=1:height                              
    for x=1:width
        dark_I(y,x) = min(Image(y,x,:));    %计算RGB取最小值后的图像dark_I
    end
end
kenlRatio = .03;                % 定义一个值为0.03的常量
krnlsz = floor(max([3, width*kenlRatio, height*kenlRatio]));    %  先计算max中三个值的最大值； 然后floor:四舍五入最大值
dark_I2 = minfilt2(dark_I, [krnlsz,krnlsz]);    %调用minfilt2计算最小值滤波
dark_I2(height,width)=0;        % 将最大行最大列置零
dark_channel=dark_I2;           % 将dark_I2赋值给dark_channel
hh=floor(width*height*0.001);   
bb=reshape(dark_channel,1,[]);  % reshape就是把指定的矩阵改变形状，但是元素个数不变，将dark_channel转为一维数组
bb=sort(bb,'descend');          % sort函数默认Mode为'ascend'为升序，sort(X,'descend')为降序排列
cc=find(bb>0,hh);       % find：bb中是否有hh个大于0的数
dd=bb(cc(hh));          % 数组bb[cc[hh]]的值
ee=dark_channel(dark_channel>dd);   
AA=(ee);
num=length(find(dark_channel>dd));
sum=0;
for k=1:num
    sum=sum+AA(k);
end
meandc=floor(sum/num);
minAtomsLight = 240;
A= min([minAtomsLight, meandc]);%计算大气光A
w0=0.95;  t0=0.3;%设置调节参数w0值
t=1-w0*(dark_channel/A);%计算透射率t
t=max(t,t0);
img_d = double(Image);
NewImage = zeros(height,width,3);
NewImage(:,:,1) = (img_d(:,:,1) - (1-t)*A)./t;%计算去雾后的R通道
NewImage(:,:,2) = (img_d(:,:,2) - (1-t)*A)./t;%计算去雾后的G通道
NewImage(:,:,3) = (img_d(:,:,3) - (1-t)*A)./t;%计算去雾后的B通道

subplot(121),imshow(uint8(Image)), title('去雾前图像');%去雾图像
subplot(122),imshow(uint8(NewImage)), title('去雾后图像');%去雾图像
imwrite(uint8(NewImage),'D:\360MoveData\Users\MarcYu\Desktop\DIP project\result\1024\0.95_H18.jpg')






function Y = minfilt2(X,varargin)
%  MINFILT2    Two-dimensional min filter
%
%     Y = MINFILT2(X,[M N]) performs two-dimensional minimum
%     filtering on the image X using an M-by-N window. The result
%     Y contains the minimun value in the M-by-N neighborhood around
%     each pixel in the original image. 
%     This function uses the van Herk algorithm for min filters.
%
%     Y = MINFILT2(X,M) is the same as Y = MINFILT2(X,[M M])
%
%     Y = MINFILT2(X) uses a 3-by-3 neighborhood.
%
%     Y = MINFILT2(..., 'shape') returns a subsection of the 2D
%     filtering specified by 'shape' :
%        'full'  - Returns the full filtering result,
%        'same'  - (default) Returns the central filter area that is the
%                   same size as X,
%        'valid' - Returns only the area where no filter elements are outside
%                  the image.
%
%     See also : MAXFILT2, VANHERK
%

% Initialization
[S, shape] = parse_inputs(varargin{:});

% filtering
Y = vanherk(X,S(1),'min',shape);
Y = vanherk(Y,S(2),'min','col',shape);

function [S, shape] = parse_inputs(varargin)
shape = 'same';
flag = [0 0]; % size shape

for i = 1 : nargin
   t = varargin{i};
   if strcmp(t,'full') && flag(2) == 0
      shape = 'full';
      flag(2) = 1;
   elseif strcmp(t,'same') && flag(2) == 0
      shape = 'same';
      flag(2) = 1;
   elseif strcmp(t,'valid') && flag(2) == 0
      shape = 'valid';
      flag(2) = 1;
   elseif flag(1) == 0
      S = t;
      flag(1) = 1;
   else
      error(['Too many / Unkown parameter : ' t ])
   end
end

if flag(1) == 0
   S = [3 3];
end
if length(S) == 1;
   S(2) = S(1);
end
if length(S) ~= 2
   error('Wrong window size parameter.')
end
end
end

%%

% img = imread('C:\Users\yijing sun\Desktop\P1\db\IEI2019\H1.jpg');
% img = rgb2gray(img);
% figure(1);
% subplot(2, 1, 1);
% imshow(img);
% title('Raw Image');
% gamma_H = 2;
% gamma_L = 0.25;
% c = 0.25;
% D0 =500;
% f = double(img);
% f = log(f + 1);%取指数
% F = fft2(f);%傅里叶变换
% F=fftshift(F);%频谱搬移
% [height, width] = size(F);
% %设计一个同态滤波器
% H = HomomorphicFiltering(gamma_H, gamma_L, c, D0, height, width);
% g=H.*F;%同态滤波
% g = ifft2(ifftshift(g));%频谱搬移,傅里叶逆变换
% g = exp(g)-1;
% g = real(g);
% %拉伸像素值
% new_img = Expand(g);
% subplot(2,1,2);
% imshow(new_img);
% title('Homomorphic Filtered Image(D0 = 100)');
% 
% 
% function new_img = Expand( img ) 
%     [height, width] = size(img);
%     max_pixel = max(max(img));
%     min_pixel = min(min(img));
%     new_img=zeros(height,width);
%     for i = 1 : height
%         for j = 1 : width
%             new_img(i, j) = 255 * (img(i, j) - min_pixel) / (max_pixel - min_pixel);
%         end
%     end
%     new_img = uint8(new_img);
% end
% 
% 
% function H = HomomorphicFiltering( gamma_H, gamma_L, c, D0, height, width )
%     for i = 1 : height
%         x = i - (height / 2);
%         for j = 1 : width
%             y = j - (width / 2);
%             H(i, j) = (gamma_H - gamma_L) * (1 - exp(-c * ((x ^ 2 + y ^ 2) / D0 ^ 2))) + gamma_L;
%         end
%     end
% end



