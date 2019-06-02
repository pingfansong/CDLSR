%%
% =========================================================================
% Coupled Sparse Coding (CSC) with L0 regularization, using OMP
% ========================================================================
% 
% The source codes are used to do image super-resolution for one modality with another different modality for guidance.
% 
% The source codes are freely available for research and study purposes.
% 
% 02/04/2018 % Coupled Sparse Coding (CSC) with L0 regularization, using OMP. 
% 
% 
% Please cite:
% ------------
% P. Song, X. Deng, J. F. Mota, N. Deligiannis, P. L. Dragotti, M. R. Rodrigues, "Multimodal Image Super-resolution via Joint Sparse Representations induced by Coupled Dictionaries", arXiv preprint arXiv:1709.08680 (2017).
% 
% P. Song, J. F. Mota, N. Deligiannis, and M. R. Rodrigues, "Coupled dictionary learning for multimodal image super-resolution", in IEEE Global Conf. Signal Inform. Process. IEEE, 2016, pp. 162¡§C166.
% 
% 
% Codes written & compiled by:
% ----------------------------
% Pingfan Song
% Electronic and Electrical Engineering,
% University College London
% uceeong@ucl.ac.uk

function [outputCSC, varargout] = CSC(data, paramsCSC, varargin)

	outputCDL = varargin{1};

	% parameters
	N = paramsCSC.N ;
	K = paramsCSC.K ;	
	s_c = paramsCSC.s_c ;
	s_x = paramsCSC.s_x ;
	s_y = paramsCSC.s_y ;


	if isfield(paramsCSC, 'SigmaNoise')
		SigmaNoise = paramsCSC.SigmaNoise ;

	else
		SigmaNoise = 0 ;
	end
	gain = 1.15 ;


	Psi_cxLR = outputCDL.Psi_cxLR;
	Psi_xLR = outputCDL.Psi_xLR;
	Psi_cx = outputCDL.Psi_cx;
	Psi_x = outputCDL.Psi_x;
	Psi_cy = outputCDL.Psi_cy;
	Psi_y = outputCDL.Psi_y;

	D = [Psi_cxLR; Psi_cy];
	normMat = diag(1./sqrt(sum(D.^2)) ) ; D = D*normMat; 
	Psi_cxLR = Psi_cxLR*normMat; Psi_cy = Psi_cy*normMat; 

	paramOMP.eps= (sqrt(N) * SigmaNoise/255 * gain)^2;  % squared norm of the residual
	paramOMP.L=s_c; % not more than xx non-zeros coefficients
	paramOMP.numThreads = -1; % number of threads
	Z_cfound=mexOMP(data, D, paramOMP);		

	X_lowR = data(1:N, :) - Psi_cxLR*Z_cfound;
	Y_highR = data((1:N)+N, :) - Psi_cy*Z_cfound;
	paramOMP.L=s_x; 
	Z_xfound = mexOMP(X_lowR, Psi_xLR, paramOMP); 

	if sum( sum(Psi_y.^2) == 0) 
		colZero = find(sum(Psi_y.^2) == 0); 
		Psi_y(:, colZero) = sqrt(1/size(Psi_y,1)); 
		Psi_y = Psi_y./ repmat(sqrt(sum(Psi_y.^2)) , [size(Psi_y ,1), 1]) ;  
	end
	paramOMP.L=s_y; 
	Z_yfound = mexOMP(Y_highR, Psi_y, paramOMP);

	% Recovery ends. Compute recovered testing dataset.
	X_rec = Psi_cx*Z_cfound + Psi_x*Z_xfound;
	Y_rec = Psi_cy*Z_cfound + Psi_y*Z_yfound;
	
	outputCSC.X_rec = X_rec;
	outputCSC.Y_rec = Y_rec;
	
	outputCSC.Z_cfound = Z_cfound;
	outputCSC.Z_xfound = Z_xfound;
	outputCSC.Z_yfound = Z_yfound;
	
end
	
	
	
	
	
	
	
	