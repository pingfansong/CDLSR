%%
% =========================================================================
% Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries
% ========================================================================
% 
% The codes are used to perform multistage coupled dictionary learning on multimodal images, 
% e.g. low-resolution and high-resolution multi-spectral images as target modality with 
% corresponding registered high-resolution RGB images as guidance modality. 
% The learned coupled dictionary can be used to enhance the low-resolution target modality with 
% the aid of high-resolution guidance modality.
% 
% The source codes are freely available for research and study purposes.
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
addpath('./utils')
addpath('./Dicts')

fprintf('Coupled dictionary learning from Infrared and RGB images. \n');
global XL X Y data XLcell Xcell Ycell
for KK = 3
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
	K = 1024; % The number of atoms.
	N = 64; % The length of one atom.
	
	MAX_ITER =400; % Total iteration number.
	trainnum = 50000; % patches from each image
	blocksize = [sqrt(N),sqrt(N)]; % [8, 8]
	stepsize = [1,1];  % the step of extracting image patches. If stepsize < blocksize, there exists overlapping aeras among adjacent patches.
	variance_Thresh = 0.04; % Discard those patch pairs with too small variance.
	upscale = 4;
	weights = 1; % weights for SI
	AnchNo = K; % number of anchored atoms.
	Nei = 2048; % 2048 ; % neighborhood size
	ImgeNum = 20; % number of images used for training
	
% 	% fast test
% 	MAX_ITER = 40; 
% 	trainnum = 10000;
% 	K = 64; 
% 	AnchNo = K; % number of anchored atoms.
% 	Nei = 10 ; % neighborhood size
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
	
% --------------------------------------------------------
%% generate image pathes and load images
	directoryX = '../data/TrainImages_Multispectral'; 
	directoryY = '../data/TrainImages_Multispectral'; 
	patternX = '*.png';
	patternY = '*.bmp';

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
%% Dictionary learning			
	

%% first layer  
	fprintf('Stage 1; ======================\n');
	% produce training samples
	
	% generate low-resolution images
	XLcell = cell(size(Xcell));
	for i = 1: numel(Xcell)
			X_tempLR = imresize( Xcell{i}, 1/upscale, 'bicubic');  
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

	outputCDL_Cell{1} = outputCDL;
	
	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
	
	outputCSR_Cell{1} = outputCSR;


%% second layer 
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

	outputCDL_Cell{2} = outputCDL;
	

	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
	
	outputCSR_Cell{2} = outputCSR;
	
%% third layer 
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

	outputCDL_Cell{3} = outputCDL;

	% reconstruction
	[outputCSR ] = Reconstruction(outputCDL, paramsCSR);
	outputCSR_Cell{3} = outputCSR;
	
	%% SAVE the dictionaries

	SIZE = ['_D',num2str(size(X,1)),'x',num2str(K)];
	MaxIter = ['_Iter', num2str(MAX_ITER)];
	TrainSize = ['_T', num2str(T)];
	Scale = ['_Scale', num2str(upscale)];

	current_date = date;
	DATE = ['_Date',current_date];

	FILENAME = ['CDL_MS', SIZE, MaxIter,TrainSize, Scale, DATE];

	save([FILENAME,'.mat'], 'paramsCDL', 'paramsCSR',  'outputCDL_Cell', 'outputCSR_Cell');
	
	
	% show results
	i=1 ;
	X_low = imresize(Xcell{i}, 1/upscale, 'bicubic');
	% generate interpolated image;
	interpolated = im2uint8( imresize(X_low, upscale, 'bicubic') );     % bicubic, bilinear
	
	figure;
	subplot(2,2,1); imagesc(Xcell{i} );	colormap gray; title('high-resolution X');	axis off;
	subplot(2,2,2); imagesc(Ycell{i} );	colormap gray;	title('high-resolution Y'); axis off;
	subplot(2,2,3); imagesc(X_low);	colormap gray; 	title('low-resolution');	axis off; 	
	subplot(2,2,4); imagesc(outputCSR{i}.ImgRec.X);	colormap gray; 	title('Estimation');	axis off; 	
	

end


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


%% draw figures
addpath('../export_fig')
plot_enable = 1;
FigFontName = 'Times New Roman';
FigFontSize = 10;
SaveFig = 0 ; % save figure or not
FigFormatCell = { '.eps', '.fig'};


if plot_enable == 1

	% show trained dictionaries
	N = paramsCDL.N;
	K = paramsCDL.K;
	Dict = outputCDL.Dict ;
	blocksize = paramsCDL.blocksize;

		
% column and row positions of four small dictionaries in the whole dictionary. 
% 	the whole dictionary = 
% 	[Psi_cx,   Psi_x,            zeros(N,K) ; ...
% 	Psi_cy,    zeros(N,K),    Psi_y];
	ind1 = 1: K;
	ind2 = (K+1) : 2*K;
	ind3 = (2*K +1) : 3*K;

	ind1_row = 1: N;
	ind2_row = (N + 1) : 2*N;

	Psi_cxLR = outputCDL(1).Psi_cxLR ;
	Psi_xLR = outputCDL(1).Psi_xLR ;
	Psi_cx = outputCDL(1).Psi_cx ;
	Psi_x = outputCDL(1).Psi_x ;
	Psi_cy = outputCDL(1).Psi_cy ;
	Psi_y = outputCDL(1).Psi_y ;
	upscale = paramsCDL.upscale;
	
% 	[~, index] = sort( sum(Psi_cxLR.^2) );
	[~, index] = sort( sum(Psi_cy.^2), 'descend' );
	Psi_cxLR = Psi_cxLR(:,index);
	Psi_cx = Psi_cx(:,index);
	Psi_cy = Psi_cy(:,index);
	
	[~, index] = sort( sum(Psi_x.^2) );
	Psi_xLR = Psi_xLR(:,index);
	Psi_x = Psi_x(:,index);
	
	[~, index] = sort( sum(Psi_y.^2) );
	Psi_y = Psi_y(:,index);	

	dictimg1 = SMALL_showdict(Psi_cxLR,blocksize,round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg2 = SMALL_showdict(Psi_xLR,blocksize, round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg3 = SMALL_showdict(Psi_cx,blocksize,	round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg4 = SMALL_showdict(Psi_x,blocksize, round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg5 = SMALL_showdict(Psi_cy,blocksize, round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');  
	dictimg6 = SMALL_showdict(Psi_y,blocksize, round(sqrt( K )), round(sqrt( K )) ,'lines','highcontrast');
	
	% Display dictionary atoms as image patch
	Hcf = figure ;
	Hca = gca ;

	subplot(3,2,1)
	imagesc(dictimg1);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cxLR}')
% 	set(gca,'position',[0.05 0.6 0.4 0.3]);

	subplot(3,2,2)
	imagesc(dictimg2);colormap(gray);axis off; axis image;
	title('Dict Psi\_{xLR}')
% 	set(gca,'position',[0.55 0.6 0.4 0.3]);

	subplot(3,2,3)
	imagesc(dictimg3);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cx}')
% 	set(gca,'position',[0.05 0.3 0.4 0.3]);

	subplot(3,2,4)
	imagesc(dictimg4);colormap(gray);axis off; axis image;
	title('Dict Psi\_{x}')
% 	set(gca,'position',[0.55 0.3 0.4 0.3]);

	subplot(3,2,5)
	imagesc(dictimg5);colormap(gray);axis off; axis image;
	title('Dict Psi\_{cy}')
% 	set(gca,'position',[0.05 0.05 0.4 0.3]);

	subplot(3,2,6)
	imagesc(dictimg6);colormap(gray);axis off; axis image;
	title('Dict Psi\_{y}')
% 	set(gca,'position',[0.55 0.05 0.4 0.3]);

	pause(0.02);
	
	set(gcf, 'Color', 'w'); % make the background to be white.
	FigPosition = [0 0 400 600];
	FigName = ['Dicts'];
	SetFigure_MultAxes(Hcf, Hca, FigPosition, FigFontName, FigFontSize, SaveFig, FigName, FigFormatCell) ;

	
end

printf('done!')


	
