%%
% =========================================================================
% Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries
% ========================================================================
% 
% The codes are used to perform multistage coupled dictionary learning on multimodal images, 
% e.g. low-resolution and high-resolution near-infrared images as target modality with 
% corresponding registered high-resolution RGB images as guidance modality. 
% The learned coupled dictionary can be used to enhance the low-resolution target modality with 
% the aid of high-resolution guidance modality.
% 
% 
% The codes are freely available for research and study purposes.

% params: contains following fields.
% 	- blocksize % pach size, e.g. [8, 8].
%	- fixed_num % 1 (default) : extract fixed number of patches from each image, according to trainnum. 0: extract patches according to step size.
% 	- trainnum % number of patches extracted from each image, e.g. 1000.
% 	- stepsize % the step of extracting image patches, e.g. [4,4]. If stepsize < blocksize, there exists overlapping aeras among adjacent patches.
%	- filter_flag % 0 (default) : Only remove mean from each patches. 1: Filtering patches with 4 filters. Remove mean only for Xh and Yh.
% 	- variance_Thresh % those patch pairs with too small variance will be discarded.
% 
% 
% Please cite:
% ------------
% P. Song, X. Deng, J. F. C. Mota, N. Deligiannis, P. Dragotti and M. Rodrigues, "Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries," in IEEE Transactions on Computational Imaging. 
% doi: 10.1109/TCI.2019.2916502
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8715417&isnumber=6960042
% 
% P. Song, J. F. Mota, N. Deligiannis, and M. R. Rodrigues, "Coupled dictionary learning for multimodal image super-resolution", in IEEE Global Conf. Signal Inform. Process. IEEE, 2016, pp. 162-166.
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

fprintf('Multistage coupled dictionary learning (with neighbourhood regression) from Infrared and RGB images. \n');
global XL X Y data XLcell Xcell Ycell
for SigmaNoise = [ 0 ] % [0, 12 ]
% parameters
	% sparsity constraints for OMP
	s_c = 4;
	s_x = 2;
	s_y = 2;	

	% lagrange multiplier for Lasso
	lambda_cxy = 0.1; 
	lambda_x = 0.05;
	lambda_y = 0.05;


	% Dictionary learning parameters
	Modality = 'RGB_NIR';
	K = 256; % The number of atoms.
	N = 64; % The length of one atom.
	
	MAX_ITER =400; % Total iteration number.
	trainnum = 50000; % 13000; % patches from each image
	blocksize = [sqrt(N),sqrt(N)]; % [8, 8]
	stepsize = [1,1];  % the step of extracting image patches. If stepsize < blocksize, there exists overlapping aeras among adjacent patches.
	variance_Thresh = 0.04; % 0.02; % Discard those patch pairs with too small variance.
	upscale = 4; % 6;
	weights = 1; % weights for SI
	AnchNo = K; % number of anchored atoms.
	Nei = 2048 ; % neighborhood size
	ImgeNum = 10; % number of images used for training
	
% 	% fast train
% 	MAX_ITER =100; % Total iteration numbe
% 	K = 64 ; %128; % 64; 
% 	AnchNo = K; % number of anchored atoms.
% 	Nei = 1024 ; % neighborhood size
% 	trainnum = 13000; % 13000; % patches from each image
% 	ImgeNum = 2; % number of images used for training
	
%% set parameters	
	paramsCDL.K = K ;
	paramsCDL.N = N ;
	paramsCDL.S = {s_c, s_x, s_y} ;
	paramsCDL.lambda = {lambda_cxy, lambda_x, lambda_y} ;
	paramsCDL.MAX_ITER = MAX_ITER ;
	paramsCDL.trainnum = trainnum ;
	paramsCDL.blocksize = blocksize ;
	paramsCDL.stepsize = stepsize ;
	paramsCDL.variance_Thresh = variance_Thresh ;
	paramsCDL.upscale = upscale;
	paramsCDL.weights = weights;	
	paramsCDL.AnchNo = AnchNo ;
	paramsCDL.Nei = Nei ;
	paramsCDL.SigmaNoise = SigmaNoise;
	
% --------------------------------------------------------
%% generate image pathes and load images
	directoryX = '../data/Train_RGB_NIR'; 
	directoryY = '../data/Train_RGB_NIR'; 
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
	% produce training samples
	
	% generate low-resolution images
	XLcell = cell(size(Xcell));
	for i = 1: numel(Xcell)
			X_tempLR = imresize( Xcell{i}, 1/upscale, 'bicubic');  
			
			% add noise to the low resolution version.
			X_tempLR = X_tempLR + SigmaNoise/255*randn(size(X_tempLR));	
			
			XLcell{i} = imresize(X_tempLR, size(Xcell{i}), 'bicubic');	
	end
	
	[X, XL, Y] = GenTrainSamples( Xcell, XLcell, Ycell, paramsCDL); 

	% Discard those patch pairs with too small variance.
	X_index = (sum(X.^2, 1) > variance_Thresh);
	Y_index = (sum(Y.^2, 1) > variance_Thresh);
	XY_index = X_index | Y_index;
	X = X(:, XY_index);
	Y = Y(:, XY_index);
	XL = XL(:, XY_index);
	Y = weights.*Y; % adjust the weights for SI.

	T = size(X,2); % training data size
	paramsCDL.T = T ;
	
	paramsCSR = paramsCDL;  % CSR param
	paramsCSR.fixed_num = 0; % extract patches with fixed step, instead of fixed number.
	

	% training
	[outputCDL, ~] = CDL(paramsCDL) ;	
	
	%compute CDL projectors
	[outputCDL] = CDLproj( outputCDL, paramsCDL) ;

% 	load('CDLproj_Stage1.mat' , 'outputCDL')  ; % uncomment to load previous trained dicts.

	outputCDL_Cell{1} = outputCDL;
	
	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
	
% % uncomment following codes to load previous reconstructed training images.
% 	FileName = 'CSR_Stage1_Train.mat';	
% 	load(FileName, 'outputCSR');  

	outputCSR_Cell{1} = outputCSR;


%% second stage 
	fprintf('Stage 2; ======================\n');
	% produce training samples
	XLcell = {}; 
	for i = 1 : numel(outputCSR)
		XLcell{i,1} = im2double( outputCSR{i}.ImgRec.X );
	end
	
	[X, XL, Y] = GenTrainSamples( Xcell, XLcell, Ycell, paramsCDL); 

	% Discard those patch pairs with too small variance.
	X_index = (sum(X.^2, 1) > variance_Thresh);
	Y_index = (sum(Y.^2, 1) > variance_Thresh);
	XY_index = X_index | Y_index;
	X = X(:, XY_index);
	Y = Y(:, XY_index);
	XL = XL(:, XY_index);
	Y = weights.*Y; % adjust the weights for SI.
	
	% training
	[outputCDL, ~] = CDL(paramsCDL) ;	
	
	% compute CDL projectors
	[outputCDL] = CDLproj( outputCDL, paramsCDL) ;

% 	load('CDLproj_Stage2.mat' , 'outputCDL')  ; % uncomment to load previous trained dicts.

	outputCDL_Cell{2} = outputCDL;
	
	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
		
% % uncomment following codes to load previous reconstructed training images.
% 	FileName = 'CSR_Stage2_Train.mat';	
% 	load(FileName, 'outputCSR');  

	outputCSR_Cell{2} = outputCSR;
	
%% third stage 
	fprintf('Stage 3; ======================\n');
	% produce training samples
	XLcell = {}; 
	for i = 1 : numel(outputCSR)
		XLcell{i,1} = im2double( outputCSR{i}.ImgRec.X );
	end
	
	[X, XL, Y] = GenTrainSamples( Xcell, XLcell, Ycell, paramsCDL); 

	% Discard those patch pairs with too small variance.
	X_index = (sum(X.^2, 1) > variance_Thresh);
	Y_index = (sum(Y.^2, 1) > variance_Thresh);
	XY_index = X_index | Y_index;
	X = X(:, XY_index);
	Y = Y(:, XY_index);
	XL = XL(:, XY_index);
	Y = weights.*Y; % adjust the weights for SI.
	
	% training
	[outputCDL, ~] = CDL(paramsCDL) ;	
	
	% compute CDL projectors
	[outputCDL] = CDLproj( outputCDL, paramsCDL) ;

% 	load('CDLproj_Stage3.mat' , 'outputCDL')  ; % uncomment to load previous trained dicts.

	outputCDL_Cell{3} = outputCDL;
	
	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
	outputCSR_Cell{3} = outputCSR;
	
% % uncomment following codes to load previous reconstructed training images.
% 	FileName = 'CSR_Stage3_Train.mat';	
% 	load(FileName, 'outputCSR');  

	
	%% SAVE the dictionaries

	SIZE = ['_D',num2str(size(X,1)),'x',num2str(K)];
	MaxIter = ['_Iter', num2str(MAX_ITER)];
	TrainSize = ['_T', num2str(T)];
	Scale = ['_Scale', num2str(upscale)];
	Noise = ['_Noise', num2str(SigmaNoise)];
	current_date = date;
	DATE = ['_Date',current_date];

	FILENAME = ['CDL_NR', Noise, SIZE, MaxIter,TrainSize, Scale, DATE];

	save([FILENAME,'.mat'], 'paramsCDL', 'paramsCSR',  'outputCDL_Cell', 'outputCSR_Cell');
	
	
	% show results
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

	save([FILENAME,'.mat'], 'paramsCDL', 'paramsCSR',  'outputCDL_Cell', 'outputCSR_Cell', 'outputCSRsum_Cell');

	disp('done! ****************************************************')


	%% show results
	results = [];
	for j = 1 :3 
		outputCSRsum = outputCSRsum_Cell{j};
		results = cat( 2, results, ...
			[outputCSRsum.MSSIM_array'; outputCSRsum.MSSIM_mean], ...
			[outputCSRsum.PSNR_array'; outputCSRsum.PSNR_mean] );

	end

	disp('****************************************************')

end

printf('done!')
disp('****************************************************')

