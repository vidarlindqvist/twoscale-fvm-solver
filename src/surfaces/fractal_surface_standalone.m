function out = fractal_surface_standalone(options)
%FRACTAL_SURFACE_STANDALONE Almqvist--Perez-Rafols generator in one file.
% No App Designer or Statistics Toolbox is required. MATLAB R2020b or later.
%
%   cfg = fractal_surface_standalone('defaults');
%   cfg.Nx=512; cfg.Ny=512; cfg.H=0.5; cfg.distribution='Gaussian';
%   out = fractal_surface_standalone(cfg);
%   mesh(out.x,out.y,out.z)
%
% Call without inputs to generate the default H=.5, Gaussian 512^2 surface.
% OPTIONS (all fields optional; unknown fields are rejected):
%   Nx,Ny             even grid dimensions, columns x / rows y [512,512]
%   distribution      'Gaussian', 'Weibull', 'Bi-Gaussian'
%   spectrum          'Self-affine' or 'Exponential'
%   H                 Hurst exponent, 0 < H < 2 [0.5]
%   beta              exponential-spectrum coefficient [0.01]
%   anisotropy        app coefficient af > 0 [1]
%   weibullShape      Weibull coefficient b > 0 [1]
%   biGaussianCoeff   app coefficient c >= 0 [4]; distribution is
%                     min(c*randn-c,randn), NOT a Gaussian mixture.
%   shortWavelength   app high-frequency cutoff, relative to reference L [1/64]
%                     zero means no upper spectral cutoff.
%   longWavelength    app 'Roll-off freq' field, relative wavelength [0.5].
%                     The original code removes low modes; NOT a plateau.
%   pipeline          'app' [default]: original post-iteration band filter;
%                     'band-first': impose short cutoff before iteration.
%   Lx,Ly             coordinate lengths [1,1]; period is N*dx, not (N-1)*dx
%   Sq,meanHeight     final RMS amplitude and mean [1,0]
%   seed              nonnegative integer [20260927]; caller RNG restored
%   maxIterations     maximum projection steps [999]
%   minSpeed          original histogram stagnation criterion [1e-3]
%   tolerance         original projection stopping threshold [1e-3]
%   plot              make diagnostic figure [true]
%   visible           'on' or 'off' ['on']
%   outputPrefix      '' => no files; otherwise full path without extension
%   exportMAT         save full result [true when prefix supplied]
%   exportXYZ         export physical X,Y,Z triplets [true]
%   exportHeight      five-line height header + z(:) [false; square pixels only]
%   exportPNG         diagnostic figure [true]
%   overwrite         allow replacing existing output files [false]
%
% SPECTRAL CONVENTION: preserved app wave-number indices
% q=sqrt((af*kx)^2+(ky/af)^2). Cutoffs are 1/longWavelength and
% 1/shortWavelength. Lx,Ly rescale coordinates; on rectangular domains the
% index-space isotropy af=1 need not be physical-space isotropy.
% TARGET C~q^(-2(H+1)); Exponential uses beta/(2*pi*(q^2+beta^2)^(3/2)).
% The initial amplitude realization is random, as in the original app.
%
% OUTPUT: z, x,y, dx,dy; zHeight/zSpectrum before final app filter;
% options, convergence history, diagnostics, original roughness parameters,
% height histogram, bearing curve and radial mean POWER spectrum.
% App filtering can change the matched height distribution. band-first
% preserves height ranks but may leave small spectral leakage. Both are
% reported; no claim of simultaneously exact finite-band PSD and histogram.
%
% Provenance: local mySurfGenApp source, copyright Andreas Almqvist 2023;
% https://www.mathworks.com/matlabcentral/fileexchange/129469
% Perez-Rafols & Almqvist, Tribology International 131 (2019) 591--604,
% doi:10.1016/j.triboint.2018.11.020.
% Refactoring 2026-09-27: options, seeded execution, periodic coordinates,
% diagnostics/export. Original generator/helper algorithms embedded below.
% Main changes: Weibull via inverse CDF (base MATLAB), explicit stop reason,
% corrected x/y dimension semantics and actual radial power diagnostic.

%% 1. Configuration and validation
cfg=defaults();
if nargin==1 && (ischar(options)||isstring(options))
    if strcmpi(options,'defaults'),out=cfg;return;end
    error('Surface:Command','Only the command ''defaults'' is supported.');
end
if nargin>0
    assert(isstruct(options)&&isscalar(options),'Surface:Options','Options must be a scalar struct.');
    names=fieldnames(options);
    for j=1:numel(names)
        assert(isfield(cfg,names{j}),'Surface:UnknownOption','Unknown option: %s',names{j});
        cfg.(names{j})=options.(names{j});
    end
end
cfg=validate(cfg);checkOutputs(cfg);
oldRNG=rng;cleanup=onCleanup(@()rng(oldRNG)); %#ok<NASGU>
rng(cfg.seed,'twister');

%% 2. Original app spectral construction
N=cfg.Nx;M=cfg.Ny;
parameter=cfg.H;if strcmp(cfg.spectrum,'Exponential'),parameter=cfg.beta;end
short=0;if strcmp(cfg.pipeline,'band-first'),short=cfg.shortWavelength;end
[~,~,ZG,C]=fractal_surf_hu_tonder_anisotropic(parameter,...
    cfg.longWavelength,short,1,1,N/M,M,N,cfg.anisotropy,cfg.spectrum);
PS=abs(fft2(ZG));PS(C==0)=0;
assert(any(PS(:)>0),'Surface:EmptySpectrum','Selected band contains no nonzero Fourier modes.');
assert(all(isfinite(PS(:))),'Surface:Spectrum','Nonfinite spectrum: check cutoffs and parameters.');

%% 3. Sample the requested height distribution
switch cfg.distribution
    case 'Gaussian'
        target=randn(M,N);
    case 'Weibull'
        b=cfg.weibullShape;
        % Shape is sufficient: later normalization sets scale and mean.
        target=-(-log(max(rand(M,N),realmin))).^(1/b);
    case 'Bi-Gaussian'
        c=cfg.biGaussianCoeff;
        target=min(c*randn(M,N)-c,randn(M,N));
end
target=target-mean(target(:));
assert(all(isfinite(target(:)))&&std(target(:))>0,'Surface:Distribution','Degenerate or nonfinite height sample.');
% Preserve original app sample scaling within iteration; normalize at output.

%% 4. Alternating spectrum and rank projections
[zh,zs,history]=fractal_surf_generator(PS,target,...
    cfg.maxIterations+1,cfg.minSpeed,cfg.tolerance);
if strcmp(cfg.pipeline,'app')
    z=fftbandfilter(zh,1,cfg.shortWavelength,1);
else
    z=zh;
end
z=normalizeHeight(z,cfg.Sq,cfg.meanHeight);
zh=normalizeHeight(zh,cfg.Sq,cfg.meanHeight);
zs=normalizeHeight(zs,cfg.Sq,cfg.meanHeight);

%% 5. Coordinates and independent diagnostics
x=(0:N-1)*cfg.Lx/N;y=(0:M-1)*cfg.Ly/M;
[X,Y]=meshgrid(x,y);
out=struct('z',z,'x',x,'y',y,'dx',cfg.Lx/N,'dy',cfg.Ly/M,...
    'zHeight',zh,'zSpectrum',zs,'options',cfg,'history',history);
out.diagnostics=diagnostics(z,PS,C,target,cfg);
out.parameters=mySurfParam(X,Y,z); % original app's descriptive parameters
[hpdf,hedges]=histcounts(z(:),80,'Normalization','pdf');
out.heightPDF=struct('height',(hedges(1:end-1)+hedges(2:end))/2,'density',hpdf);
q=linspace(0,1,min(numel(z),1001));v=sort(z(:),'descend');
out.bearingCurve=struct('materialRatio',q,'height',interp1(linspace(0,1,numel(v)),v,q));
out.provenance='Almqvist-Perez-Rafols local mySurfGenApp; standalone refactoring 2026-09-27';

%% 6. Optional plots and exports (no files by default)
fig=[];
if cfg.plot || (~isempty(cfg.outputPrefix)&&cfg.exportPNG)
    fig=plotSurface(out,cfg.visible);
end
if ~isempty(cfg.outputPrefix)
    parent=fileparts(cfg.outputPrefix);if ~isempty(parent)&&~isfolder(parent),mkdir(parent);end
    if cfg.exportMAT,save([cfg.outputPrefix '.mat'],'out');end
    if cfg.exportXYZ
        fid=fopen([cfg.outputPrefix '_xyz.txt'],'w');assert(fid>=0,'Surface:File','Cannot open XYZ file.');
        fcloseGuard=onCleanup(@()fclose(fid));
        fprintf(fid,'%% x y z; periodic samples, no duplicated endpoint\n');
        fprintf(fid,'%.17g %.17g %.17g\n',[X(:),Y(:),z(:)]');clear fcloseGuard
    end
    if cfg.exportHeight
        fid=fopen([cfg.outputPrefix '_height.txt'],'w');assert(fid>=0,'Surface:File','Cannot open height file.');
        fcloseGuard=onCleanup(@()fclose(fid));
        fprintf(fid,'%% Almqvist_PerezRafols\n%% %d\n%% %d\n%% %.17g\n%% 1\n',N,M,out.dx);
        fprintf(fid,'%.17g\n',z(:));clear fcloseGuard
    end
    if cfg.exportPNG,exportgraphics(fig,[cfg.outputPrefix '.png'],'Resolution',160);end
end
if ~cfg.plot && ~isempty(fig),close(fig);end
end

function c=defaults()
c=struct('Nx',512,'Ny',512,'distribution','Gaussian','spectrum','Self-affine',...
 'H',.5,'beta',.01,'anisotropy',1,'weibullShape',1,'biGaussianCoeff',4,...
 'shortWavelength',1/64,'longWavelength',.5,'pipeline','app',...
 'Lx',1,'Ly',1,'Sq',1,'meanHeight',0,'seed',20260927,...
 'maxIterations',999,'minSpeed',1e-3,'tolerance',1e-3,...
 'plot',true,'visible','on','outputPrefix','','exportMAT',true,...
 'exportXYZ',true,'exportHeight',false,'exportPNG',true,'overwrite',false);
end

function c=validate(c)
for f={'Nx','Ny'}
 validateattributes(c.(f{1}),{'numeric'},{'scalar','real','finite','integer','>=',8});
 assert(mod(c.(f{1}),2)==0,'Surface:Grid','Nx and Ny must be even.');
end
for f={'beta','anisotropy','weibullShape','Lx','Ly','Sq','longWavelength','tolerance'}
 validateattributes(c.(f{1}),{'numeric'},{'scalar','real','finite','positive'});
end
validateattributes(c.H,{'numeric'},{'scalar','real','finite','>',0,'<',2});
for f={'shortWavelength','biGaussianCoeff','minSpeed'}
 validateattributes(c.(f{1}),{'numeric'},{'scalar','real','finite','nonnegative'});
end
validateattributes(c.meanHeight,{'numeric'},{'scalar','real','finite'});
validateattributes(c.seed,{'numeric'},{'scalar','real','finite','integer','>=',0,'<=',2^32-1});
validateattributes(c.maxIterations,{'numeric'},{'scalar','real','finite','integer','positive'});
assert(c.longWavelength<=1&&c.shortWavelength<c.longWavelength,'Surface:Band','Require 0 <= shortWavelength < longWavelength <= 1.');
c.distribution=validatestring(c.distribution,{'Gaussian','Weibull','Bi-Gaussian'});
c.spectrum=validatestring(c.spectrum,{'Self-affine','Exponential'});
c.pipeline=validatestring(c.pipeline,{'app','band-first'});
c.visible=validatestring(c.visible,{'on','off'});
for f={'plot','exportMAT','exportXYZ','exportHeight','exportPNG','overwrite'}
 validateattributes(c.(f{1}),{'logical','numeric'},{'scalar','binary'});
end
assert(ischar(c.outputPrefix)||(isstring(c.outputPrefix)&&isscalar(c.outputPrefix)),'Surface:Prefix','outputPrefix must be text.');
c.outputPrefix=char(c.outputPrefix);
if c.exportHeight
 assert(abs(c.Lx/c.Nx-c.Ly/c.Ny)<=1e-12*max(c.Lx/c.Nx,c.Ly/c.Ny),...
 'Surface:SquarePixels','Five-line height format requires dx=dy; use XYZ or MAT otherwise.');
end
% Test band before calling the original helper (which normalizes its output).
[kx,ky]=meshgrid([0:c.Nx/2-1 -c.Nx/2:-1],[0:c.Ny/2-1 -c.Ny/2:-1]);
k=hypot(c.anisotropy*kx,ky/c.anisotropy);
mask=k>=1/c.longWavelength;
if strcmp(c.pipeline,'band-first')&&c.shortWavelength>0,mask=mask&k<=1/c.shortWavelength;end
assert(any(mask(:)),'Surface:EmptySpectrum','No supported modes in requested band.');
if strcmp(c.pipeline,'app')&&c.shortWavelength>0
 assert(any(mask(:)&hypot(kx(:),ky(:))<=1/c.shortWavelength),'Surface:EmptySpectrum','Post-filter removes every target mode.');
end
end

function z=normalizeHeight(z,sq,mu)
z=real(z);z=z-mean(z(:));s=sqrt(mean(z(:).^2));
assert(isfinite(s)&&s>0,'Surface:ZeroVariance','Surface has no finite nonzero variance.');z=z/s*sq+mu;
end

function d=diagnostics(z,PS,C,target,c)
u=(z-mean(z(:)))/c.Sq;F=fft2(u);A=PS/sqrt(sum(PS(:).^2))*numel(PS);
d=struct('mean',mean(z(:)),'Sq',sqrt(mean((z(:)-mean(z(:))).^2)),...
 'skewness',mean(u(:).^3),'kurtosis',mean(u(:).^4),'min',min(z(:)),'max',max(z(:)));
d.spectrumRelativeAmplitudeError=sqrt(mean((abs(F(A>0))./A(A>0)-1).^2));
d.outsideIterationBandVariance=sum(abs(F(C==0)).^2)/sum(abs(F(:)).^2);
t=normalizeHeight(target,1,0);d.sortedHeightRMSError=sqrt(mean((sort(u(:))-sort(t(:))).^2));
[kx,ky]=meshgrid([0:c.Nx/2-1 -c.Nx/2:-1],[0:c.Ny/2-1 -c.Ny/2:-1]);
k=hypot(kx,ky);P=abs(F/numel(F)).^2;
bins=logspace(0,log10(max(k(:))+eps),41);cent=nan(1,40);power=cent;
for j=1:40
 m=k>=bins(j)&k<bins(j+1);if any(m(:)),cent(j)=exp(mean(log(k(m))));power(j)=mean(P(m));end
end
d.radialMode=cent;d.radialMeanPower=power;d.Hfitted=NaN;
hi=min(c.Nx,c.Ny)/4;if c.shortWavelength>0,hi=min(hi,.75/c.shortWavelength);end
m=cent>=2/c.longWavelength&cent<=hi&power>0&isfinite(power);
if strcmp(c.spectrum,'Self-affine')&&c.anisotropy==1&&nnz(m)>=4
 fit=polyfit(log(cent(m)),log(power(m)),1);d.Hfitted=-fit(1)/2-1;
end
d.fitMeaning='H fit is reported only for isotropic self-affine targets; fit is a realization diagnostic.';
end

function checkOutputs(c)
if isempty(c.outputPrefix)||c.overwrite,return;end
ext={};if c.exportMAT,ext{end+1}='.mat';end
if c.exportXYZ,ext{end+1}='_xyz.txt';end
if c.exportHeight,ext{end+1}='_height.txt';end
if c.exportPNG,ext{end+1}='.png';end
for j=1:numel(ext),assert(~isfile([c.outputPrefix ext{j}]),'Surface:Exists','Output exists: %s',[c.outputPrefix ext{j}]);end
end

function fig=plotSurface(s,vis)
% Three-panel verification layout; normalization affects display only.
c=s.options; d=s.diagnostics; u=(s.z-d.mean)/d.Sq;
fig=figure('Visible',vis,'Color','w','Position',[60 60 1500 480]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(s.x/c.Lx,s.y/c.Ly,u); axis image;
set(gca,'YDir','normal'); colormap(fig,parula); colorbar;
xlabel('x/L_x'); ylabel('y/L_y');
if strcmp(c.spectrum,'Self-affine')
    title(sprintf('%s surface, H = %.3g',c.distribution,c.H));
else
    title(sprintf('%s surface, Exponential spectrum',c.distribution));
end
nexttile; histogram(u(:),80,'Normalization','pdf','FaceColor',[.3 .65 .8]); hold on;
if strcmp(c.distribution,'Gaussian')
    t=linspace(min(-5,min(u(:))),max(5,max(u(:))),400);
    plot(t,exp(-t.^2/2)/sqrt(2*pi),'k--','LineWidth',1.5);
end
xlabel('(z - mean(z)) / S_q'); ylabel('Probability density');
title(sprintf('Skew %.3f; kurtosis %.3f',d.skewness,d.kurtosis));
nexttile;
k=d.radialMode; power=d.radialMeanPower;
limit=min(c.Nx,c.Ny)/2;
if c.shortWavelength>0
    limit=min(limit,max(c.anisotropy,1/c.anisotropy)/c.shortWavelength);
end
lower=min(c.anisotropy,1/c.anisotropy)/c.longWavelength;
show=isfinite(k)&power>0&k>=lower&k<=limit;
loglog(k(show),power(show),'o-','LineWidth',1); hold on;
if isfinite(d.Hfitted)
    hi=min(c.Nx,c.Ny)/4;
    if c.shortWavelength>0, hi=min(hi,.75/c.shortWavelength); end
    fitMask=k>=2/c.longWavelength&k<=hi&power>0&isfinite(power);
    fit=polyfit(log(k(fitMask)),log(power(fitMask)),1);
    loglog(k(fitMask),exp(polyval(fit,log(k(fitMask)))),'k--','LineWidth',1.3);
    title(sprintf('Measured H = %.4f',d.Hfitted));
else
    title('Radial spectrum (H fit not applicable)');
end
grid on; xlabel('Radial Fourier mode'); ylabel('Mean power per mode');
end

%% Embedded original algorithms and original descriptive parameters

function [Zh,Zs,err] = fractal_surf_generator(PS,Zh,...
                                      max_it,min_speed,min_err)
[M,N] = size(PS);
% Scale and center the power spectrum appropiatelly
PS(1,1) = 0;
PS      = PS/sqrt(sum(PS(:).^2))*M*N;
% Scale and center the height distribution
Zh_mean = mean(Zh(:));
Zh      = Zh - Zh_mean;
Sq      = sqrt(mean(Zh(:).^2));
Zh      = Zh/Sq;

Zh_order = sort(Zh(:));
[n0,bin] = histcounts(Zh_order,'BinMethod','scott',...
                                 'Normalization','pdf');

% Initialize errors and Zh
err.H   = 1;
err.PS  = 1;
err.PS0 = 1;
Zh      = reshape(Zh,M,N);
zh      = fft2(Zh);    
while 1
  % Step 1: Fix the power spectrum by FFT filter
  phase = angle(zh);
  Zs    = real(ifft2(PS.*exp(1i*phase)));  
  % Error in height distribution
  n            = histcounts(Zs(:),bin,'Normalization','pdf');
  err.H(end+1) = sum(abs(n-n0).*(bin(2:end)-bin(1:end-1)));

  % Step 2: Fix the height distribution rank ordering
  Zh      = reshape(Zs,1,M*N);
  [~,ind] = sort(Zh);
  Zh(ind) = Zh_order;
  Zh      = reshape(Zh,M,N);
  zh      = fft2(Zh);           
  % Error in power spectrum
  Zh_abs         = abs(zh);
  err.PS(end+1)  = sqrt(mean((1-Zh_abs(PS>0)./PS(PS>0)).^2));
  err.PS0(end+1) = sqrt(mean(Zh_abs(PS==0).^2))/mean(PS(PS>0));
            
  if length(err.H) == max_it
    err.stopReason = 'maximum_iterations';
    break
  end
  if abs(err.H(end) - err.H(end-1))/max(err.H(end),eps) < min_speed
    err.stopReason = 'histogram_stagnation';
    break
  end
  if err.H(end)<min_err || ...
    (err.PS(end)<min_err && err.PS0(end)<min_err)
    err.stopReason = 'tolerance';
    break
  end
end

% Rescale and output
Zh = Zh*Sq + Zh_mean;
Zs = Zs*Sq + Zh_mean;
end



function [X,Y,Z,C] = fractal_surf_hu_tonder_anisotropic...
                                        (H,Llong,Lshort,Sq,Lx,Ly,M,N,af,PS)
% This code generates a random surface with a power density spectra
% corresponding to a self-affine fractal, C(q) ~ k^(-2*H-2). To do so, we
% use the technique presented by Hu and Tonder (1992) in Int. J. Mach.
% Tools Manufact., vol 32 No. 1/2, pp. 83-90, entiteled "Simulation of 3-D
% random rough surface by a 2-D digital filter and fourier analysis".

% The main idea behind the code is to filter a random sample with the right
% height distribution to obtain a surface with the right wave distribution
% and the right power density spectra. 

% The input variable are
% H      - Hurst exponent
% Llong  - Longest wavelength pressent
% Lshort - Shortest wavelength present
% Sq     - Sq parameter for the resultant surface
% Lx     - Length of the domain, x-dir
% Ly     - Length of the domain, y-dir
% M      - Number of nodes, y-dir
% N      - Number of nodes, x-dir

% Transform the cut-off's to wavenumber
minq = (Lx/Llong);    % Long wavelengths cutoff
maxq = (Lx/Lshort);   % Short wavelengths cutoff

% Matrix form of the wavenumbers
[qx,qy] = meshgrid(-N/2:N/2,-M/2:M/2);

% Account for anisotropy when defining the 1D wavenumber
q       = sqrt((af*qx).^2 + (qy./af).^2);

% Define the power spectral density
if strcmp(PS,'Self-affine')
    C         = q.^(-2*(H+1));
elseif strcmp(PS,'Exponential')
    C         = H./(2*pi*(q.^2+H^2).^(3/2));
end
% Apply the cut-off's
C(q<minq) = 0;
C(q>maxq) = 0;
% Rearange C to match matlabs interpretation of fft
C         = fftshift(C(1:end-1,1:end-1));

% Define the filter function as a function of C
Hf = sqrt(C);

% Gaussian random numbers generation
eta = randn(M,N);

% Filter eta to obtain a surface with the defined C(q)
Z = ifft2(Hf.*fft2(eta));

% Scale Z
Z = Z - mean(Z(:));
Z = Z/sqrt(mean(Z(:).^2));
Z = Z*Sq;

% Defint the X,Y,Z triplet
[X,Y] = meshgrid(linspace(0,Lx,N),linspace(0,Ly,M));



end

function Z = fftbandfilter(z,Llong,Lshort,Lx)

[M,N] = size(z);

% Define low- and high frequency cutoff
lowq  = (Lx/Llong);   % <=> Long wavelength cutoff (Relative)
highq = (Lx/Lshort);  % <=> Short wavelength cutoff (Relative)

% Frequency vector
[qx,qy] = meshgrid(-N/2:N/2,-M/2:M/2);
q       = sqrt(qx.^2+qy.^2);

% Set up the transfer function
H          = ones(size(q));
H(q<lowq)  = 0; % Apply long wavelength cutoff
H(q>highq) = 0; % Apply short wavelength cutoff
H          = fftshift(H(1:end-1,1:end-1)); % fftshift is required to comply
                                           % with MATLABs ordering for fft

% Filter surface (and truncate imaginary noise)
Z = real(ifft2(fft2(z).*H));


end

function Param = mySurfParam(X,Y,Z)
        % Adjusting the mean value to zero
        Z = Z - mean(Z(:));
        z = Z(:);

        % Height parameters
        Param.a  = mean(abs(z));          % Average roughness
        Param.q  = sqrt(mean(z.^2));      % rms roughness
        Param.sk = mean(z.^3)/Param.q^3;  % Skewness
        Param.ku = mean(z.^4)/Param.q^4;  % Kurtosis
        
        Param.p = max(z);
        Param.v = min(z);
        Param.z = Param.p-Param.v;
        
        % Average "max" and "min" for 5 highest and lowest values.
        zsort = sort(z);
        Param.p5 = mean(zsort(end-5+1:end));
        Param.v5 = mean(zsort(1:5));
        Param.z5 = mean(zsort(end-5+1:end))-mean(zsort(1:5));

        % Hybrid parameter(s)
        dzdy = mydiffy(Y,Z); 
        dzdx = mydiffx(X,Z);
        Param.dq = sqrt(mean(dzdy(:).^2 + dzdx(:).^2)); % rms slope 

        % Abbot curve parameters
          % Obtain cumulative function
          AB = sort(z,'Descend');
          AB = interp1(1:numel(AB),AB,linspace(1,numel(AB),1e3));
          AB = sort(unique(AB.','rows').','Descend');
          nbins = numel(AB); 
          XA = linspace(0,100,nbins);
          % Fit line with minimum slope taking 40 %  of the data
          N40   = floor(0.4*nbins);
          slope = Inf;
          NAB   = 1;
          for jj = 1:1:nbins-N40
             % fit slope 
             y = AB(jj:jj+N40);
             x = XA(jj:jj+N40);
             s = sum((x-mean(x)).*(y-mean(y)))/sum((x-mean(x)).^2);
             n = mean(y)-s*mean(x);
             % Save value if minimum
             if abs(s) < abs(slope)
                slope = s;
                inter = n;
                NAB   = jj;
             end
          end

        % Find parameters
          % Heights at intercepts
          H1 = inter;
          H2 = inter + slope*100;
          % Percentage at intercepts height
          Param.Mr1 = max(0.11,interp1(AB,XA,H1));
          Param.Mr2 = min(interp1(AB,XA,H2),100-0.11); 
          % Areas of peaks and valeys on the curve
          A1  = trapz(XA(XA<Param.Mr1),AB(XA<Param.Mr1)-H1);
          A2  = trapz(XA(XA>Param.Mr2),(H2-AB(XA>Param.Mr2)));
          % Core Sk parameters
          Param.Sk  = H1-H2;
          Param.Spk = 2*A1/Param.Mr1;
          Param.Svk = 2*A2/(100-Param.Mr2);
          % Volumes
          H10 = interp1(XA,AB,10);
          H80 = interp1(XA,AB,80);    
          Param.Vmp = trapz(XA(XA<10),AB(XA<10)-H10)/100;
          Param.Vmc = trapz(XA(XA<80),AB(XA<80)-H80)/100-Param.Vmp;
          Param.Vvv = trapz(XA(XA>80),H80-AB(XA>80))/100;
          Param.Vvc = trapz(XA(XA>10),H10-AB(XA>10))/100-Param.Vvv;
          
%{
Sk(Core height) is calculated as the difference of heights at the areal
material ratio values 0% and 100% on the equivalent line; specifically, 
it is a value obtained by subtracting the minimum height from the maximum 
height of the core surface.

Sk represents the height of the core surface
The reduced peaks are the areas that are removed by initial abrasion.
Smr1 represents the areal material ratio that divides the reduced peaks 
from the core surface.
Spk represents the average height of the reduced peaks.
The reduced valleys are the areas that hold liquid applied on the surface 
in order to improve lubricity. 
Smr2 represents the areal material ratio that divides the reduced valleys 
from the core surface.
Svk represents the average depth of the reduced valleys.
%}

end

function Fx = mydiffx(x,F,varargin)
%MYDIFFX(X,F) computes the finite difference derivative of the 
%   function F w.r.t. X.
%   
%   FX = MYDIFFX(X,F,ORDER) computes the finite difference derivative 
%   of specified ORDER of the function F w.r.t. X. 
%
%   X is a 1 x N vector and F an M x N matrix
%
%   ORDER is 1, (2) or 4. 
%
%   See also mydiffy, mydiff, myfftdiffx, myfftdiffy, myfftdiff

%   Author(s): A. Almqvist
%   Copyright 2009- Andreas Almqvist
%   $Revision: 1$  $Date: 2009/11$
%   Homemade function

% Default order is 2
order = 2;
if ~isempty(varargin)
    order = varargin{1};
end
N   = size(x,2);
if order == 1;
    j   = 2:N;
    Fx  = [(-3*F(:,1)+4*F(:,2)-F(:,3))/2, F(:,j)-F(:,j-1),...
        ]/(x(1,2)-x(1,1));
elseif order == 2;
    j   = 2:N-1;
    Fx  = [-3*F(:,1)+4*F(:,2)-F(:,3),...
        F(:,j+1)-F(:,j-1),...
        F(:,N-2)-4*F(:,N-1)+3*F(:,N) ...
        ]/2/(x(1,2)-x(1,1));
elseif order == 4;
    if N < 6;
        j   = 2:N-1;
        Fx  = [-3*F(:,1)+4*F(:,2)-F(:,3),...
            -F(:,j-1)+F(:,j+1),...
            F(:,N-2)-4*F(:,N-1)+3*F(:,N)...
            ]/2/(x(1,2)-x(1,1));
    else
        j   = 3:N-2;
        Fx  = [-25*F(:,1:2)+48*F(:,2:3)-36*F(:,3:4)+16*F(:,4:5)-3*F(:,5:6),...
            F(:,j-2)-8*F(:,j-1)+8*F(:,j+1)-F(:,j+2),...
            3*F(:,N-5:N-4)-16*F(:,N-4:N-3)+36*F(:,N-3:N-2)-48*F(:,N-2:N-1)+25*F(:,N-1:N)...
            ]/12/(x(1,2)-x(1,1));
    end
end

end

function Fy = mydiffy(y,F,varargin)
%MYDIFFY(Y,F) computes the finite difference derivative of the 
%   function F w.r.t. Y.
%   
%   FY = MYDIFFY(Y,F,ORDER) computes the finite difference derivative 
%   of specified ORDER of the function F w.r.t. Y.
%
%   Y is a M x 1 vector and F an M x N matrix
%
%   ORDER is 1, (2) or 4. 
%
%   See also mydiffx, mydiff, myfftdiffx, myfftdiffy, myfftdiff

%   Author(s): A. Almqvist
%   Copyright 2009- Andreas Almqvist
%   $Revision: 1$  $Date: 2009/11$
%   Homemade function
% Default order is 2
order = 2;
if ~isempty(varargin)
    order = varargin{1};
end

Fy = mydiffx(y',F',order)';

end

%% Original source license
% Copyright (c) 2023, Andreas Almqvist
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution
% 
% * Neither the name of  nor the names of its
%   contributors may be used to endorse or promote products derived from this
%   software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
