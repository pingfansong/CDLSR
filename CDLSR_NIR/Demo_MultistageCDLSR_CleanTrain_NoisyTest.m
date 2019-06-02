%%
% =========================================================================
% Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries
% ========================================================================
% 
% The codes are used to perform multimodal image super-resolution using 
% a group of coupled dictionaries trained on multi-modal image patches.
% Specifically, given a modality, e.g. near-infrared images, of low-resolution and a
% guidance modality, e.g. RGB images, of high-resolution, the algorithm
% enhance the low-resolution target modality with the aid of high-resolution
% guidance modality.
% 
% The codes reproduce experiments in Section V.B 2).
% The codes are freely available for research and study purposes.
% 
% 
% Please cite:
% ------------
% P. Song, X. Deng, J. F. C. Mota, N. Deligiannis, P. Dragotti and M. Rodrigues, "Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries," in IEEE Transactions on Computational Imaging. 
% doi: 10.1109/TCI.2019.2916502
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8715417&isnumber=6960042
% 
% P. Song, J. F. Mota, N. Deligiannis, and M. R. Rodrigues, "Coupled dictionary learning for multimodal image super-resolution", in IEEE Global Conf. Signal Inform. Process. IEEE, 2016, pp. 162¨C166.
% 
% 
% Codes written & compiled by:
% ----------------------------
% Pingfan Song
% Electronic and Electrical Engineering,
% University College London
% uceeong@ucl.ac.uk
% 
% =========================================================================

clear; 
addpath('./ksvd/ksvdbox/private_ccode');
addpath('./SPAMS/build')
addpath(genpath('./utils'))
addpath('./Dicts')

fprintf('Super-resolution for infrared images with RGB images for guidance. \n');
global XL X Y data XLcell Xcell Ycell

for SigmaNoise = [4] % 2:2:12
% parameters
	
	% load dictionaries.
	CDLname = 'CDL_Stage123_RGB_NIR_D64x1024_Scale4_Date30-Mar-2018.mat';
	load( CDLname );
	
	upscale = paramsCDL.upscale ;
	K = paramsCDL.K ; % number of atoms.
	N = paramsCDL.N; % length of one atom.
	paramsCDL.SigmaNoise = SigmaNoise ; % standard deviation for images with pixel range [0,255].
	
	ImgeNum = 1; % number of testing images to be loaded
	outputCSR_Cell = {}; % store output results
	
	upscale = paramsCDL.upscale  ; 
	weights = paramsCDL.weights;

	% location, width used to crop image.
	croploc = [1,1];
	blocksize = [sqrt(N), sqrt(N)]; % the size of each image patch.
	stepsize = [1,1];  % the step of extracting image patches. If stepsize < stepsize, there exists overlapping aeras among adjacent patches.
	s_c = 12;
	s_x = 1;
	s_y = 1;	
	Ssum_rec = s_c+s_x+s_y;  % threshold for OMP
	variance_Thresh_SR = 0.005; % 0.1; % 0.005; % threshold for smooth patches

	% lagrange multiplier for Lasso
	lambda_cxy = 0.1; 
	lambda_x = 0.05;
	lambda_y = 0.05;
	
	paramsCSR = paramsCDL;
	paramsCSR.N = N;
	paramsCSR.K = K;	
	paramsCSR.croploc =croploc;
	paramsCSR.blocksize = blocksize;
	paramsCSR.stepsize = stepsize;
	paramsCSR.s_c = s_c;
	paramsCSR.s_x = s_x;
	paramsCSR.s_y = s_y;
	paramsCSR.lambda = {lambda_cxy, lambda_x, lambda_y} ;
	paramsCSR.upscale = upscale;
	paramsCSR.variance_Thresh_SR = variance_Thresh_SR;
	paramsCSR.SigmaNoise = SigmaNoise;
	paramsCSR.fixed_num = 0; % extract patches with fixed step, instead of fixed number.


% --------------------------------------------------------
%% generate image pathes and load images
	directoryX = '../data/Test_RGB_NIR'; 
	directoryY = '../data/Test_RGB_NIR'; 
	patternX = '*nir.tiff';
	patternY = '*rgb.tiff';

	Xfilepaths = dir( fullfile(directoryX, patternX) );
	Yfilepaths = dir( fullfile(directoryY, patternY) );

	Xfilepaths = Xfilepaths(1: ImgeNum);
	Yfilepaths = Yfilepaths(1: ImgeNum);
	
	if numel(Xfilepaths) ~= numel(Xfilepaths)	
		disp('Warning: The number of X images is not equal to the number of Y images!');
	end
	
	XpathCell = cell(numel(Xfilepaths), 1);
	YpathCell = cell(numel(Yfilepaths), 1);
	for i = 1:numel(Xfilepaths)
		XpathCell{i} = fullfile(directoryX, Xfilepaths(i).name);
		YpathCell{i} = fullfile(directoryY, Yfilepaths(i).name);
	end
	
	% load images
	Xcell = load_images( XpathCell );
	Ycell = load_images( YpathCell );
	
	% crop each images
	Xcell = modcrop(Xcell, upscale);
	Ycell = modcrop(Ycell, upscale);
	
	paramsCDL.XpathCell = XpathCell;
	paramsCDL.YpathCell = YpathCell;	

% --------------------------------------------------------	
%% Multistage CDLSR	

%% first stage  
	fprintf('Stage 1; ======================\n');
		
	% generate low-resolution images as input
	XLcell = cell(size(Xcell));
	for i = 1: numel(Xcell)
			X_tempLR = imresize( Xcell{i}, 1/upscale, 'bicubic');  
			
			% add noise to the low resolution version.
			X_tempLR = X_tempLR + SigmaNoise/255*randn(size(X_tempLR));	
			
			XLcell{i} = imresize(X_tempLR, size(Xcell{i}), 'bicubic');	
	end
	
	
	outputCDL = outputCDL_Cell{1} ;
	
	% reconstruction
	TSTART = tic;
	[outputCSR ] = Reconstruction_OMP(outputCDL, paramsCSR);
	t1 = toc(TSTART);
	
	outputCSR_Cell{1} = outputCSR;


%% second stage 
	fprintf('Stage 2; ======================\n');
	% Use the estimated HR output from the previous stage as current input.
	XLcell = {}; 
	for i = 1 : numel(outputCSR)
		XLcell{i,1} = im2double( outputCSR{i}.ImgRec.X );
	end
	
	outputCDL = outputCDL_Cell{2} ;
	
	% reconstruction
	[outputCSR ] = Reconstruction_OMP(outputCDL, paramsCSR);
	t2 = toc(TSTART);
	
	outputCSR_Cell{2} = outputCSR;
	
%% third stage 
	fprintf('Stage 3; ======================\n');
	% Use the estimated HR output from the previous stage as current input.
	XLcell = {}; 
	for i = 1 : numel(outputCSR)
		XLcell{i,1} = im2double( outputCSR{i}.ImgRec.X );
	end
	
	outputCDL = outputCDL_Cell{3} ;
	
	% reconstruction
	[outputCSR ] = Reconstruction_OMP(outputCDL, paramsCSR);
	t3 = toc(TSTART);
	outputCSR_Cell{3} = outputCSR;
	
	%% SAVE the dictionaries

% 	SIZE = ['_D',num2str(N),'x',num2str(K)];
	Scale = ['_Scale', num2str(upscale)];
	Noise = ['_Noise', num2str(SigmaNoise)];
	current_date = date;
	DATE = ['_Date',current_date];

	FILENAME = ['CSR_CleanTrain_Test', Noise, Scale, DATE];

	save([FILENAME,'.mat'], 'paramsCSR', 'outputCSR_Cell');
	
	
	%% show results
	i=1 ;
	X_low = imresize(Xcell{i}, 1/upscale, 'bicubic');
	
	% add noise to the low resolution version.
	X_low = X_low + SigmaNoise/255*randn(size(X_low));	
			
	% generate interpolated image;
	interpolated = im2uint8( imresize(X_low, upscale, 'bicubic') );     % bicubic, bilinear
	
	figure;
	subplot(2,2,1); imagesc(Xcell{i} );	colormap gray; title('high-resolution X');	axis off;
	subplot(2,2,2); imagesc(Ycell{i} );	colormap gray;	title('high-resolution Y'); axis off;
	subplot(2,2,3); imagesc(X_low);	colormap gray; 	title('low-resolution');	axis off; 	
	subplot(2,2,4); imagesc(outputCSR{i}.ImgRec.X);	colormap gray; 	title('Estimation');	axis off; 	
	
	%% summarize the results and compute the mean
	outputCSRsum_Cell = {};
	for j = 1 :3 
		outputCSR = outputCSR_Cell{j};
		PSNR_array = []; RMSE_array = []; MSSIM_array = [];
		for i = 1 : numel(outputCSR)
			PSNR_array(i) = outputCSR{i}.PSNRall.PSNR_X ;
			MSSIM_array(i) = outputCSR{i}.mssim.X ;
		end
		outputCSRsum_Cell{j}.PSNR_array = PSNR_array ;
		outputCSRsum_Cell{j}.MSSIM_array = MSSIM_array ;
		outputCSRsum_Cell{j}.PSNR_mean = mean(PSNR_array) ;
		outputCSRsum_Cell{j}.MSSIM_mean = mean(MSSIM_array) ;
	end

	save([FILENAME,'.mat'], 'paramsCSR', 'outputCSR_Cell', 'outputCSRsum_Cell');


	%% show results
	results = [];
	for j = 1 :3 
		outputCSRsum = outputCSRsum_Cell{j};
		results = cat( 2, results, ...
			[outputCSRsum.MSSIM_array'; outputCSRsum.MSSIM_mean], ...
			[outputCSRsum.PSNR_array'; outputCSRsum.PSNR_mean] );

	end

end

printf('done!')
disp('****************************************************')


