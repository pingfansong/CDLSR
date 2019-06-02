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
% guidance modality. There is no neighbourhood regressioin.
% 
% 
% The codes are freely available for research and study purposes.
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

function [outputCSR, varargout] = Reconstruction(outputCDL, paramsCSR, varargin)
global XL X Y data XLcell Xcell Ycell

% reconstruction
	outputCSR = cell(length(Xcell) , 1) ;	
	weights = paramsCSR.weights ;
	blocksize = paramsCSR.blocksize  ;
	stepsize = paramsCSR.stepsize ;
	upscale = paramsCSR.upscale;
	variance_Thresh_SR = paramsCSR.variance_Thresh_SR; % threshold for smooth patches
	
	info_all = [];
	for t = 1: length(Xcell) 
		
		disp('------------------------------')
		info=sprintf('Processing image %d of total %d ... \n', t, length(Xcell));

		X_test = Xcell{t} ;
		Y_test = Ycell{t} ;
		cropwidth = size( Xcell{t} ); 		
		

		% Load training image and construct training dataset.	
		[~, X_test_vecLR, Y_test_vec, DC] = GenTrainSamples( Xcell(t), XLcell(t), Ycell(t), paramsCSR); 
		Y_test_vec = weights.*Y_test_vec; % add weights on the side information	
% 		data = [X_test_vecLR; Y_test_vec];
% 		clear X_test_vecLR Y_test_vec
		 
		T = size(data, 2);
		AtomLength = blocksize(1) * blocksize(2); % the length of atoms and 
		X_rec = zeros(AtomLength, T); % recovered X;
		
% 		fprintf('Patches = %d x %d. ', AtomLength, T);

		% first round coupled sparse coding
		CSC = @CSC_L0_OMP ; % % CSC with basic version, no NR.
		
		% store away those patches with small variance.
		X_test_vecLR0 = X_test_vecLR; 
		Y_test_vec0 = Y_test_vec;
		indCol = find(sum(X_test_vecLR.^2, 1) >= variance_Thresh_SR) ;
		X_test_vecLR = X_test_vecLR(:, indCol);
		Y_test_vec = Y_test_vec(:, indCol) ;
		data = [X_test_vecLR; Y_test_vec];
		
		tic
% 		[X_rec] = CSC(paramsCSR, outputCDL);
		
		[outputCSC] = CSC( data , paramsCSR, outputCDL);
		X_rec = outputCSC.X_rec;

		timeElapsed = toc;
	
		% add smooth patches into
		X_test_vecLR0(:, indCol) = X_rec;
		
		X_rec = X_test_vecLR0;
		
		%%
		% After reconstructing the centered HR patches, we add the mean back to them. 
		X_rec = X_rec + repmat(DC.Xl, size(X_rec,1), 1);
		X_rec_im = col2imstep(X_rec, cropwidth, blocksize, stepsize);	

		clear X_rec

		% average over the overlapping blocks of the separated signals
		cnt = countcover(cropwidth,blocksize,stepsize);
		for i = 1:size(cnt,1)
			for j = 1:size(cnt,2)
				if cnt(i,j) == 0
					cnt(i,j) = 1;
				end
			end
		end
		X_rec_im = X_rec_im./cnt; 
		X_rec_im(X_rec_im > 1) = 1;
		X_rec_im(X_rec_im < 0) = 0;
		ImgRec.X = im2uint8(X_rec_im);
		
	%% comput PSNR, SSIM
		[PSNR_X, SNR_X] = psnr(X_rec_im, X_test); % or [PSNR_X, SNR_X] = psnr(ImgX,RefX,peakval); 

		info = sprintf('%s PSNR_X = %.4f; \n', info, PSNR_X);
		disp(info); 		
		info_all = [info_all, info];

		PSNRall.PSNR_X = PSNR_X;

		[mssim.X, ssim_map{2}] = ssim(im2uint8(X_rec_im), im2uint8(X_test)) ;
		
		outputCSR{t}.ImgRec = ImgRec;
		outputCSR{t}.PSNRall = PSNRall;
		outputCSR{t}.mssim = mssim;
		outputCSR{t}.info = info_all;		

	end
	
% 	varargout{1} = [] ;
	
end
	
	
	
	