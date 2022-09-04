%{
 Restoration Image Script
 File: restore_image_script.m
 Written by: Abdullah Emir Gogusdere, 
 August 19, 2022 
 Contact e-mail, abdullah.gogusdere@metu.edu.tr
%}

% This script benefits from the "Slant Edge Script" by Patrick Granton. 
% For more details, refer to the reference at the end of the file.

close all; clearvars;

isotropicpixelspacing = 0.03; 
% isotropic detector pixel spacing in mm, (i.e. pixel pitch).  
% set this value to your detector

pixel_subdivision = 0.08; 
% try to keep pixel_subdivision between 0.03 -> 0.15 
% since it provides a good trade-off between sampling uniformity and noise,
% especially 0.1 subpixel bin spacing.

bin_pad = 0.0001;
% It adds a small space to include all values to the histogram 
% of edge spread distances.

boundplusminus = 4;
% boundplusminus is a variable that is used to crop a small section of the
% edge in order to used to find the subpixel interpolated edge position.

boundplusminus_extra = 1;
% boundplusminus_extra incorperates addition pixel values near the edge to 
% include in the binned histogram.

image_path = "test_image.JPG";
[fPath, fName, fExt] = fileparts(image_path);

switch lower(fExt)
  case '.tif'
    image = imread(image_path);
    image = image(:,:,1:3);
    image = rgb2gray(image);  
%     For noise testing    
%     image = imnoise(image,'gaussian');
%     image = imgaussfilt(image, 2);
    image_full = image;
    image_full_raw = image;
  case {'.jpg','.png'}
    image = rgb2gray(imread(image_path));
%     For noise testing    
    image = imnoise(image,'gaussian');
    image = imgaussfilt(image, 2);
    image_full = image;
    image_full_raw = image;
  otherwise
    error('Unexpected file extension: %s', fExt);
end

% For selecting the best step-edge region interactively.
% figure("Name", "Select a rectangular region, and double click on it.");
% [image,rect] = imcrop(image);
% rect

rect = [434.5100  203.5100   15.9800  132.9800];
crop_col_leftpoint = round(rect(1)); crop_row_leftpoint = round(rect(2));
crop_collength = round(rect(3)); crop_rowlength = round(rect(4));

image = image_full(crop_row_leftpoint:crop_row_leftpoint+crop_rowlength...
    ,crop_col_leftpoint:crop_col_leftpoint+crop_collength);

[img_rowlength, img_columnlength] = size(image);

level = graythresh(image); % Determinig the threshold of the canny edge detector
 
% Detect edge and orientation
BW_edge_raw = edge(double(image),'canny', level);

% Locate edge positions
[y_row_pos, x_column_pos] = find(BW_edge_raw==1);

% Fit edge positions
P = polyfit(x_column_pos,y_row_pos,1); % mx + b = y

% determine rough edge angle to determine orientation
angle_radians = atan(P(1));

if abs(angle_radians) > pi/4 % i.e. edge is vertical
    start_row = boundplusminus_extra;
    end_row = img_rowlength - boundplusminus_extra;
    BW_mask = false(img_rowlength,img_columnlength);
    
    roi = zeros(end_row-start_row+1, 3);
    counter = 1;
    
    for i = start_row:end_row
        [BW_y_row, BW_x_col] = find(BW_edge_raw(i,:)==1);

        index_start = min(BW_x_col)-boundplusminus;
        index_end = max(BW_x_col)+boundplusminus;
        if index_start <= 0
            index_start = 1;
        end
        if index_end > img_columnlength
            index_end = img_columnlength;
        end

        BW_mask(i,index_start:index_end) = 1;
        roi(counter,:) = [i,index_start,index_end];
        counter = counter + 1;
    end
    
    [y_row_pos, x_column_pos, values] = find(image.*uint8(BW_mask));

else % the edge is horizontal
    start_col = boundplusminus_extra;
    end_col = img_columnlength - boundplusminus_extra;
    BW_mask = false(img_rowlength,img_columnlength);
    
    roi = zeros(end_col-start_col+1, 3);
    counter = 1;
    for i = start_col:end_col
        [BW_y_row, BW_x_col] = find(BW_edge_raw(:,i)==1);
        index_start = min(BW_y_row)-boundplusminus;
        index_end = max(BW_y_row)+boundplusminus;
        if index_start <= 0
            index_start = 1;
        end
        if index_end > img_rowlength
            index_end = img_rowlength;
        end
        BW_mask(index_start:index_end,i) = 1;
        roi(counter,:) = [index_start,index_end,i];
        counter = counter + 1;
    end

    % Locate edge positions
    [y_row_pos, x_column_pos, values] = find(image.*uint8(BW_mask)); 
    
    % Fit edge positions
    P = polyfit(x_column_pos,y_row_pos,1); 
    angle_radians = atan(P(1));
end

% Visualizing the area going to be processed
figure;
subplot(1,2,1);
imshow(image);
title("The image of the edge")
subplot(1,2,2);
imshow(uint8(BW_mask)*255);
title("the edge area going to be processed")

% transforming the coordinates to the new coordinate system
transformed_angle_radians = -(pi/2 - abs(angle_radians))*(angle_radians > 0) + ...
    (pi/2 - abs(angle_radians))*(angle_radians < 0);
transformed_edge_position = [x_column_pos*cos(transformed_angle_radians) + y_row_pos*sin(transformed_angle_radians),...
                     x_column_pos*-sin(transformed_angle_radians) + y_row_pos*cos(transformed_angle_radians)];

% offsetting and sorting the values regard to their x-coordinates
mean_trans_edge_position = mean(transformed_edge_position(:,1));
sorted_edge_position_plus_value = sortrows(cat(2, transformed_edge_position(:,1)-mean_trans_edge_position, double(values)));

array_positions_of_edge = sorted_edge_position_plus_value(:,1);
array_values_of_edge = sorted_edge_position_plus_value(:,2);

% Determine bin spacing 
topEdge = max(array_positions_of_edge) + bin_pad + pixel_subdivision;
botEdge = min(array_positions_of_edge) - bin_pad;
binEdges = botEdge:pixel_subdivision:topEdge;
numBins = length(binEdges) - 1;
binPositions = binEdges(1:end-1) + 1/2*pixel_subdivision;
binMean = zeros(1,length(numBins)); % preallocation

% enumarating the values according to their bins
which_bin = discretize(array_positions_of_edge, binEdges);

for i = 1:numBins
    flagBinMembers = (which_bin == i);
    binMembers = array_values_of_edge(flagBinMembers);
    binMean(i) = mean(binMembers);	
end

ESF_raw = sorted_edge_position_plus_value(:,2);
xESF_raw = sorted_edge_position_plus_value(:,1);

ESF_bin = binMean(2:numBins - 1); % Eliminate first, second and last array position
xESF_bin = binPositions(2:numBins - 1); % same as above comment

% removing NaN values
ESF_bin_nonan = ESF_bin(logical(1-isnan(ESF_bin)));
xESF_bin_nonan = xESF_bin(logical(1-isnan(ESF_bin)));

% filling the missing data
xESF_bin = xESF_bin_nonan;
ESF_bin = interp1(xESF_bin_nonan,ESF_bin_nonan,xESF_bin,'pchip');


fit_ESF_P = polyfit(xESF_raw,ESF_raw,9);
fit_xESF = linspace(min(xESF_raw),max(xESF_raw),length(xESF_raw));
fit_ESF = polyval(fit_ESF_P,fit_xESF);

figure;
plot(xESF_raw, ESF_raw, xESF_bin, ESF_bin, fit_xESF, fit_ESF);
title('The Binned ESF of the edge')
xlabel('distance along the rotated axis in pixels')
ylabel('Gray level (intensity)')
legend("Raw ESF", "Only binned", "9th degree poly fitted ESF")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Differentiating the ESF to get LSF
LSF_raw = abs(diff(ESF_raw));
LSF_bin = abs(diff(ESF_bin));
LSF = abs(diff(fit_ESF));

% Removing the last element since diff() outputs in the one difference size of
% original array
xLSF_raw = xESF_raw(1:end-1);
xLSF_bin = xESF_bin(1:end-1);
xLSF = fit_xESF(1:end-1);

% Choosing two points to determine the region for applying the Hamming window
f = figure("Name", "Select two points on the graph of the Binned LSF");
plot(xLSF_bin,LSF_bin);
[px,~] = getpts(f);
x1 = find(xLSF_bin > px(1)-0.1 & xLSF_bin < px(1)+0.1);
x2 = find(xLSF_bin > px(2)-0.1 & xLSF_bin < px(2)+0.1);
close(f);

temp = LSF_bin; % To save the original binned ESF data

% Applying hamming window method to the LSF that uses binned ESF
LSF_hamming = LSF_bin(x1:x2);
LSF_hamming = LSF_hamming .* hamming(length(LSF_hamming))';
LSF_ham = LSF_bin;
LSF_ham(x1:x2) = LSF_hamming;

LSF_bin = temp;

% Normalize the LSF
LSF = LSF/sum(LSF);
LSF_raw = LSF_raw/sum(LSF_raw);
LSF_bin = LSF_bin/sum(LSF_bin);
LSF_ham = LSF_ham/sum(LSF_ham);

figure;
plot(xLSF_raw,LSF_raw,...
     xLSF_bin,LSF_bin,...
     xLSF_bin,LSF_ham,...
     xLSF,LSF);
title('The LSF of the Edge')
legend('Raw LSF', 'LSF with Binned ESF', 'Hammed LSF', 'Fitted LSF')
xlabel('distance along the rotated axis in pixels')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 4096; % sample number
 
% generate the frequency axis
if mod(N,2)==0
    q = -N/2:N/2-1; % N even
else
    q = -(N-1)/2:(N-1)/2; % N odd
end

% Convert the spatial domain to frequency domain
Fs = 1/(pixel_subdivision*isotropicpixelspacing); % sampling rate in samples per mm
freq = q*Fs/N;
 
%%% Generating the MTF %%%
MTF_raw = (abs(fftshift(fft(LSF_raw,N)))/max(max(abs(fftshift(fft(LSF_raw,N))))))';
MTF_bin = abs(fftshift(fft(LSF_bin,N)))/max(max(abs(fftshift(fft(LSF_bin,N)))));
MTF_ham = abs(fftshift(fft(LSF_ham,N)))/max(max(abs(fftshift(fft(LSF_ham,N)))));
MTF = abs(fftshift(fft(LSF,N)))/max(max(abs(fftshift(fft(LSF,N)))));

all_MTF = [MTF_raw;MTF_bin;MTF_ham;MTF];
str_MTF = ["Raw MTF", "MTF with binned ESF", "MTF with hammed LSF", "fitted MTF"];

figure;
plot(freq,MTF_raw,freq,MTF_bin,freq,MTF_ham,freq,MTF);

title('The MTF of the Edge')
legend('Raw MTF', 'MTF with binned ESF', 'MTF with hammed LSF', 'fitted MTF')
xlabel('frequency distribution (cycles/mm)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, img_col] = size(image_full);

figure;
shortaxis = round(length(MTF)/2)+1:round(length(MTF)/2)+img_col;

plot(freq(shortaxis),MTF_raw(shortaxis),freq(shortaxis),MTF_bin(shortaxis),...
    freq(shortaxis),MTF_ham(shortaxis),freq(shortaxis),MTF(shortaxis));

title('The MTF of the Edge (ZOOMED)')
legend('Raw MTF', 'MTF with binned ESF', 'MTF with hammed LSF', 'fitted MTF')
xlabel('frequency distribution (cycles/mm)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.052; % manually selected
image_array = [];

figure;
imshow(image_full_raw);
title("Original Image")

for i=1:size(all_MTF,1)

    image_full = image_full_raw; % Resetting image_full

    %%%% Preparing the wiener filter %%%%
    
    MTF = all_MTF(i,:);

    % Getting rid of the symmetric part of the MTF
    shortaxis = round(length(MTF)/2)+1:round(length(MTF)/2)+img_col;
    MTF = MTF(shortaxis);
    
    magnitude_MTF = MTF.*conj(MTF);
    wiener = magnitude_MTF./(magnitude_MTF + gamma)./MTF;
    
    % Applying the wiener filter to the all rows of the image
    F = fftshift(fft2(image_full));
    restored_image = ifft2(ifftshift(F.*wiener)); 
    restored_image = sqrt(restored_image.*conj(restored_image));
    
    min_value = double(min(min(image_full_raw)));
    max_value = double(max(max(image_full_raw)));
    restored_image = uint8(rescale(restored_image,min_value,max_value));
    image_full = restored_image;
    image_array = cat(3,image_array,image_full);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Measuring the MTF of the restored images %%%%

all_MTF_reconstructed = zeros(size(image_array,3),N);

for x=1:size(image_array,3)
    image_full = image_array(:,:,x);
   
    image = image_full(crop_row_leftpoint:crop_row_leftpoint+crop_rowlength...
        ,crop_col_leftpoint:crop_col_leftpoint+crop_collength);
    
    [img_rowlength, img_columnlength] = size(image);
    
    level = graythresh(image); % Determine the threshold of the canny edge detector
     
    % Detect edge and orientation
    BW_edge_raw = edge(double(image),'canny', level);
    
    % Locate edge positions
    [y_row_pos, x_column_pos] = find(BW_edge_raw==1);
    
    % Fit edge positions
    P = polyfit(x_column_pos,y_row_pos,1); % mx + b = y
    
    % determine rough edge angle to determine orientation
    angle_radians = atan(P(1));
    
    if abs(angle_radians) > pi/4 % i.e. edge is vertical
        start_row = boundplusminus_extra;
        end_row = img_rowlength - boundplusminus_extra;
        BW_mask = false(img_rowlength,img_columnlength);
        
        roi = zeros(end_row-start_row+1, 3);
        counter = 1;
        
        for i = start_row:end_row
            [BW_y_row, BW_x_col] = find(BW_edge_raw(i,:)==1);
    
            index_start = min(BW_x_col)-boundplusminus;
            index_end = max(BW_x_col)+boundplusminus;
            if index_start <= 0
                index_start = 1;
            end
            if index_end > img_columnlength
                index_end = img_columnlength;
            end
    
            BW_mask(i,index_start:index_end) = 1;
            roi(counter,:) = [i,index_start,index_end];
            counter = counter + 1;
        end
        
        [y_row_pos, x_column_pos, values] = find(image.*uint8(BW_mask));
    
    else % the edge is horizontal
        start_col = boundplusminus_extra;
        end_col = img_columnlength - boundplusminus_extra;
        BW_mask = false(img_rowlength,img_columnlength);
        
        roi = zeros(end_col-start_col+1, 3);
        counter = 1;
        for i = start_col:end_col
            [BW_y_row, BW_x_col] = find(BW_edge_raw(:,i)==1);
            index_start = min(BW_y_row)-boundplusminus;
            index_end = max(BW_y_row)+boundplusminus;
            if index_start <= 0
                index_start = 1;
            end
            if index_end > img_rowlength
                index_end = img_rowlength;
            end
            BW_mask(index_start:index_end,i) = 1;
            roi(counter,:) = [index_start,index_end,i];
            counter = counter + 1;
        end
    
        % Locate edge positions
        [y_row_pos, x_column_pos, values] = find(image.*uint8(BW_mask)); 
        
        % Fit edge positions
        P = polyfit(x_column_pos,y_row_pos,1); 
        angle_radians = atan(P(1));
    end
    % transforming the coordinates to the new coordinate system
    transformed_angle_radians = -(pi/2 - abs(angle_radians))*(angle_radians > 0) + ...
        (pi/2 - abs(angle_radians))*(angle_radians < 0);
    transformed_edge_position = [x_column_pos*cos(transformed_angle_radians) + y_row_pos*sin(transformed_angle_radians),...
                         x_column_pos*-sin(transformed_angle_radians) + y_row_pos*cos(transformed_angle_radians)];
    
    % offsetting and sorting the values regard to their x-coordinates
    mean_trans_edge_position = mean(transformed_edge_position(:,1));
    sorted_edge_position_plus_value = sortrows(cat(2, transformed_edge_position(:,1)-mean_trans_edge_position, double(values)));
    
    array_positions_of_edge = sorted_edge_position_plus_value(:,1);
    array_values_of_edge = sorted_edge_position_plus_value(:,2);
    
    ESF_raw = sorted_edge_position_plus_value(:,2);
    xESF_raw = sorted_edge_position_plus_value(:,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Differentiating the ESF to get LSF
    LSF_raw = abs(diff(ESF_raw));
    
    % Removing the last element since diff() outputs in the one difference size of
    % original array
    xLSF_raw = xESF_raw(1:end-1);
    
    % Normalize the LSF
    LSF_raw = LSF_raw/sum(LSF_raw);
     
    %%% Generating the MTF %%%
    MTF_raw = (abs(fftshift(fft(LSF_raw,N)))/max(max(abs(fftshift(fft(LSF_raw,N))))))';
    all_MTF_reconstructed(x,:) = MTF_raw;
end

%%% Plotting the reconstructed images with their MTFs %%%%

[~, img_col] = size(image_full);
shortaxis = round(length(MTF_raw)/2)+1:round(length(MTF_raw)/2)+100;    

for i=1:size(all_MTF_reconstructed,1)
    figure;
    subplot(1,2,1);
    imshow(image_full_raw);
    title("Original Image")
    subplot(1,2,2);
    imshow(image_array(:,:,i));
    title(sprintf("Restored Image using '%s'", str_MTF(i)))
end

for i=1:size(all_MTF_reconstructed,1)
    figure;
    plot(freq(shortaxis),all_MTF_reconstructed(i,shortaxis));
    title(sprintf('The MTF of the Edge (ZOOMED) (%s)',str_MTF(i)))
    xlabel('frequency distribution (cycles/mm)')
end

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Reference:
P. Granton, "Slant Edge Script," MathWorks, 2 September 2010. 
[Online]. Available: https://www.mathworks.com/matlabcentral/mlc-downloads
/downloads/submissions/28631/versions/1/previews/MATHWORKS_MTF/MTFscript.m
/index.html. [Accessed 19 August 2022].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
