function [Xh, Xl, Yh, varargout] = GenTrainSamples(Xcell, XLcell, Ycell, params, varargin)
%%
% ====================================== ===================================
% Generate image patches from multimodal images.
% Xcell : all the high-resolution images of target modality.
% XLcell : all the low-resolution images of target modality.
% Ycell : all the high-resolution images of guidance modality.
% params: contains following fields.
% 	- blocksize % pach size, e.g. [8, 8].
%	- fixed_num % 1 (default) : extract fixed number of patches from each image, according to trainnum. 0: extract patches according to step size.
% 	- trainnum % number of patches extracted from each image, e.g. 1000.
% 	- stepsize % the step of extracting image patches, e.g. [4,4]. If stepsize < blocksize, there exists overlapping aeras among adjacent patches.
%	- filter_flag % 0 (default) : Only remove mean from each patches. 1: Filtering patches with 4 filters. Remove mean only for Xh and Yh.

% =========================================================================


% K = params.K ; % The number of atoms.
% N = params.N ; % The length of one atom.

trainnum = params.trainnum ; % {500, 1000,5000,10000}
blocksize = params.blocksize ; % {[8,8],[8,8],[8,8]}	
upscale = params.upscale;
stepsize = params.stepsize ;

% 	save_flag = params.save_flag ;
if isfield(params, 'filter_flag')
	filter_flag = params.filter_flag ;
else
	filter_flag = 0 ;
end


if isfield(params, 'fixed_num')
	fixed_num = params.fixed_num ;
else
	fixed_num = 1 ; % extract patches with fixed number.
end


% extract patches and vectorize them to construct training dataset.	
Xh = []; % High resolution X
Yh = []; % High resolution Y
Xl = []; % Low resolution X.
Xm = []; % middle resolution resulting from bicubic interpolation.

ImgNum= size(Xcell , 1) * size(Xcell, 2);

if filter_flag == 0
	% No filtering.Remove mean for all.
	for i = 1: ImgNum
		X_temp = Xcell{i}; 
		Y_temp = Ycell{i};
		X_tempLR = XLcell{i}; 

% 			figure; i=1;
% 			subplot(1,3,1); imagesc(Xcell{i} );	colormap gray; title('high-res X');	axis off;
% 			subplot(1,3,2); imagesc(XLcell{i});	colormap gray; 	title('low-res');	axis off; 	
% 			subplot(1,3,3); imagesc(Ycell{i} );	colormap gray;	title('high-res Y'); axis off;

		if fixed_num
		%% extract patches with fixed number
				p = ndims(Y_temp); % Number of dimensions.
				ids = cell(p,1); % indices
				[ids{:}] = reggrid(size(Y_temp)-blocksize+1, trainnum, 'eqdist');
				Y_a = sampgrid(Y_temp,blocksize,ids{:});
				X_a = sampgrid(X_temp,blocksize,ids{:});		
				X_aLR = sampgrid(X_tempLR,blocksize,ids{:});

				Xh = [Xh X_a];
				Yh = [Yh Y_a];			
				Xl = [Xl X_aLR];

				TT(i) = size(Xh,2); % training data size
		else
		%% extract patches with fixed step
				% extract patches with fixed step and vectorize them and construct dataset.
				X_temp_vec = []; % vectorized patches from  image X;
				Y_temp_vec = []; % vectorized patches from  image Y;
				X_temp_vecLR = []; % vectorized patches from LR  image X;

				cropwidth = size(X_temp);
				batch_counter = 0;
				for j = 1 : stepsize(2) : cropwidth(2)-blocksize(2)+1 % One column is one batch.
					batch_counter = batch_counter + 1;
					% Rearrange the current batch of image blocks/ patches into corresponding columns block by block . 
					blocks_X = im2colstep(X_temp(:, j:j+blocksize(1)-1),blocksize,stepsize);
					blocks_Y = im2colstep(Y_temp(:, j:j+blocksize(1)-1),blocksize,stepsize);
					blocks_XLR = im2colstep(X_tempLR(:, j:j+blocksize(1)-1),blocksize,stepsize);

					X_temp_vec = [X_temp_vec, blocks_X];
					Y_temp_vec = [Y_temp_vec, blocks_Y];	
					X_temp_vecLR = [X_temp_vecLR, blocks_XLR];

				end

				Xh = [Xh X_temp_vec];
				Yh = [Yh Y_temp_vec];			
				Xl = [Xl X_temp_vecLR];
		end
	end
	
	% remove the mean (the dc component from each patch)
% 		dc_Xh = mean(Xh);
	dc_Yh = mean(Yh);
	dc_Xl = mean(Xl);

	Yh = Yh - repmat(dc_Yh, size(Yh, 1), 1);
	Xl = Xl - repmat(dc_Xl, size(Xl, 1), 1);		
	% Xh should remove mean referring to Xl;
	Xh = Xh - repmat(dc_Xl, size(Xh, 1), 1);

	% The following is Yang and Yi Ma's way as dc_Xh is very close to dc_Xl from interpolation.
	% 		Xh = Xh - repmat(dc_Xh, size(Xh, 1), 1);  

	DC.Yh = dc_Yh;
	DC.Xl = dc_Xl;

	varargout{1} = DC;

elseif filter_flag == 1
	% Filtering with 4 filters. Remove mean only for Xh and Yh.
	for i = 1: ImgNum
		X_temp = Xcell{i}; 
		Y_temp = Ycell{i};

		X_tempLR_real = imresize(X_temp, 1/upscale, 'bicubic');
		X_tempLR = imresize(X_tempLR_real, size(X_temp), 'bicubic');	
		X_tempMR = X_tempLR ;   

		% compute the first and second order gradients
		hf1 = [-1,0,1];
		vf1 = [-1,0,1]';

		lImG11 = conv2(X_tempLR, hf1,'same');
		lImG12 = conv2(X_tempLR, vf1,'same');

		hf2 = [1,0,-2,0,1];
		vf2 = [1,0,-2,0,1]';

		lImG21 = conv2(X_tempLR,hf2,'same');
		lImG22 = conv2(X_tempLR,vf2,'same');

		% add all of these filtered low-res images
		lImG11_2 = lImG11 + lImG12;
		lImG21_2 = lImG21 + lImG22;
		X_tempLR = lImG11_2 + lImG21_2;


		p = ndims(Y_temp); % Number of dimensions.
		ids = cell(p,1); % indices
		[ids{:}] = reggrid(size(Y_temp)-blocksize+1, trainnum, 'eqdist');
		Y_a = sampgrid(Y_temp,blocksize,ids{:});
		X_a = sampgrid(X_temp,blocksize,ids{:});
		X_aLR = sampgrid(X_tempLR,blocksize,ids{:});
		X_aMR = sampgrid(X_tempMR,blocksize,ids{:});


		Xh = [Xh X_a];
		Yh = [Yh Y_a];
		Xl = [Xl X_aLR];		
		Xm = [Xm X_aMR];	


		ShowFigure = 0;
		if ShowFigure
			figure; 
			subplot(2,2,1);	imagesc(X_tempLR_real ); colormap gray;
			title('real low-res'); 			axis off
			set(gca, 'position', [0, 0.5, 0.45, 0.45 ]);

			subplot(2,2,2); imagesc(X_tempMR); colormap gray;
			title('interp low-res'); axis off
			set(gca, 'position', [0.55, 0.5, 0.45, 0.45 ]);

			subplot(2,2,3);	imagesc(X_tempLR ); colormap gray;
			title('filtered low-res'); axis off
			set(gca, 'position', [0, 0, 0.45, 0.45 ]);

			subplot(2,2,4);	imagesc(X_temp); colormap gray;
			title('high-res'); axis off
			set(gca, 'position', [0.55, 0, 0.45, 0.45 ]);

			set(gcf, 'position', [100,100,500,500]);
		end
	end

	% remove the mean (the dc component from each patch)
	% 	dc_Xh = mean(Xh);
	dc_Yh = mean(Yh);
	dc_Xm = mean(Xm);

	Yh = Yh - repmat(dc_Yh, size(Yh, 1), 1);	
	% Xh should remove mean referring to Xm;
	Xh = Xh - repmat(dc_Xm, size(Xh, 1), 1);

	DC.Yh = dc_Yh;
	DC.Xl = dc_Xm;
	
	varargout{1} = DC;
	
end

% % Discard those patch pairs with too small variance.
% Xnorm2 = sum(Xh.^2, 1);
% X_index = (Xnorm2 > variance_Thresh);
% Ynorm2 = sum(Yh.^2, 1);
% Y_index = (Ynorm2 > variance_Thresh);
% 
% Xlnorm2 = sum(Xl.^2, 1);
% 
% XY_index = X_index | Y_index;
% 
% Xh = Xh(:, XY_index);
% Yh = Yh(:, XY_index);
% 
% Xl = Xl(:, XY_index);
%}


T = size(Xh,2); % training data size


% contrast normalization and whitening
FlagNormVar = 0; % normalize the variance or not
FlagWhite = 0; % whitening or not

%-----------------------------
% contrast normalization to obtain unit variance.
if FlagNormVar
	varstd = sqrt(sum(X.^2)); % standare variance
	ita = 0.01 ;
	index = varstd < ita ;
	varstd(index) = ita ;

	X = X./ repmat(varstd, [size(X,1),1]) ;
end

%-----------------------------
% whitening. Make the variance matrix close to the identity matrix.
if FlagWhite
	X = whiten(X);
end
	
% --------------------------------------------------------	

% 	if save_flag ==1 
% 		TrainSize = ['_TSize',num2str(N),'x',num2str(T)];
% 		Scale = ['_Scale', num2str(upscale)];
% 
% 	% 			current_date = date;
% 	% 			DATE = ['_Date',current_date];
% 
% 		FILENAME = ['TrainData', TrainSize, Scale];
% 
% 		save([FILENAME,'.mat'], ...
% 			'params', 'Xh','Yh','Xl')
% 	
% 	end
	
end










