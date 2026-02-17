%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
%
%
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions
%   are met:
%
%     Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the
%     documentation and/or other materials provided with the
%     distribution.
%
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%

% cascade_MIMO_signalProcessing.m
%
% Top level main test chain to process the raw ADC data. The processing
% chain including adc data calibration module, range FFT module, DopplerFFT
% module, CFAR module, DOA module. Each module is first initialized before
% actually used in the chain.

clearvars
close all

PLOT_ON = 1; % 1: turn plot on; 0: turn plot off
LOG_ON = 1; % 1: log10 scale; 0: linear scale
% numFrames_toRun = 10; %number of frame to run, can be less than the frame saved in the raw data
SAVEOUTPUT_ON = 0;
PARAM_FILE_GEN_ON = 1;
DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP = 0 ; % Will make things slower
dataPlatform = 'TDA2'

%% get the input path and testList
pro_path = getenv('CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO');
input_path = strcat(pro_path,'\main\cascade\input\');
testList = strcat(input_path,'testList.txt');
%path for input folder
fidList = fopen(testList,'r');
testID = 1;
function plotWaveletForChirps(adcData, outputDir, frame_index_val)
    % Ensure output directory exists
    adcData = reshape(adcData, size(adcData,1), size(adcData,2), size(adcData,3) * size(adcData,4)); % Convert to 3D

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Define sampling frequency
    Fs = 8e6; % 8 MHz

    % Total number of chirps
    totalChirps = size(adcData, 2);

    % Loop over chirps in groups of 16
    chirpsPerGroup = 128;
    numGroups = ceil(totalChirps / chirpsPerGroup);

    for groupIdx = 1:numGroups
        % Determine chirps in the current group
        startChirp = (groupIdx - 1) * chirpsPerGroup + 1;
        endChirp = min(groupIdx * chirpsPerGroup, totalChirps);

        % Select data for the current chirp group
        selectedData = adcData(:, startChirp:endChirp, 1); % Assuming antenna index 1

        % Flatten the data and convert to complex double
        flattenedData = complex(double(real(selectedData(:))), double(imag(selectedData(:))));

        % Define time vector
        t = (0:length(flattenedData) - 1) / Fs;

        % Perform Continuous Wavelet Transform (CWT)
        [cfs_real, f_real] = cwt(real(flattenedData), Fs);
        [cfs_imag, f_imag] = cwt(imag(flattenedData), Fs);

        % Compute magnitude of wavelet coefficients
        cfs_magnitude = abs(cfs_real + 1i * cfs_imag);

        % Third plot: Magnitude of wavelet coefficients
       figure;
        h_mag = pcolor(t, f_real, abs(cfs_magnitude).^2);
        set(h_mag, 'EdgeColor', 'none');
        colormap jet;
        %colorbar;

        % Remove x and y axis labels and title
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title(sprintf('CWT Power (Group %d)', groupIdx));

       
        % ---- Tick marks (measures) ----
        xticks(linspace(0, 2e-4, 5));          % 0, 0.5e-4, 1e-4, 1.5e-4, 2e-4
        yticks(0:1e6:5e6);                     % 0 to 5 MHz in 1 MHz steps
        
        % Make ticks readable (show MHz on y, microseconds on x)
        ax = gca;
        ax.XAxis.Exponent = 0;
        ax.YAxis.Exponent = 0;

        box on;
        grid on;
        % Set limits
        ylim([0 5e6]);
        xlim([0 2e-4]);

        % Overlay contour lines
        hold on;
        contour(t, f_real, abs(cfs_magnitude).^2, 'LineWidth', 1, 'LineColor', 'k'); % Black contour lines
        hold off;

        % Save the plot without axis and labels
        saveas(gcf, fullfile(outputDir, sprintf('imag%d_wavelet%d_magnitude.png',frame_index_val,groupIdx)));

        % Close figure to avoid open figure windows
        close(gcf);
              
    end

    disp('Wavelet plots (real, imaginary, and magnitude with contours) for all chirps have been generated and saved.');
end
function plotWaveletForRangeBins(rangeFFT)
    % Define wavelet and widths for CWT
    numSamplePerChirp = 256;    % Number of samples per chirp
    numChirpPerLoop = 12;       % Number of chirps per loop
    numLoops = 128;              % Number of loops per frame
    numRXPerDevice = 4;         % Number of receiving channels per device
    numDevices = 4;             % Number of devices in the cascade (if needed)
    sample_rate = 8e6;          % Sampling rate (in Hz)
    wavelet = 'cmor2.5-1.0';  % Complex Morlet wavelet
    disp(size(rangeFFT));
    first_chirp_first_loop = squeeze(rangeFFT(5, :, 1, 1));
    first_chirp_first_loop = first_chirp_first_loop(:);   % force vector
    % Define time-sampled array (in seconds) based on sampling rate
    Tc  = (45) * 1e-6;  % 45 us
    PRF = 1 / Tc;                              % 22.22 kHz
    
    time_sampled = (0:length(first_chirp_first_loop)-1) / PRF; 
    [cwtmatr, freqs_new] = cwt(first_chirp_first_loop, sample_rate);
    % Take absolute value of complex result to get magnitude
    cwtmatr = abs(cwtmatr(:,:,1));
    disp(size(cwtmatr(:,:,1)));
    % Plot the scaleogram (CWT result)
    figure;
    pcolor(time_sampled, freqs_new, cwtmatr);
    shading interp;  % Smoothing the plot
    set(gca, 'YScale', 'log');  % Set logarithmic scale for frequency axis
    
    xlabel('Time (s)');  % Label time axis in seconds
    ylabel('Frequency (Hz)');  % Label frequency axis in Hz
    title('CWT (Scaleogram) of First Chirp Signal');
    colormap('jet');
    colorbar;
    outputDir = 'C:\Users\asus\Documents\Projects\FYP\DSP\Matlab\RadarSignalProcessingMatlab\4chip_cascade_MIMO_example';
    saveas(gcf, fullfile(outputDir, sprintf('range%d_wavelet_slow.png', 4)));

end
while ~feof(fidList)
    
    %% get each test vectors within the test list
    % test data file name
    dataFolder_test = fgetl(fidList);    
   
    %calibration file name
    dataFolder_calib = fgetl(fidList);
    
    %module_param_file defines parameters to init each signal processing
    %module
    module_param_file = fgetl(fidList);
    
     %parameter file name for the test
    pathGenParaFile = [input_path,'test',num2str(testID), '_param.m'];
    %important to clear the same.m file, since Matlab does not clear cache
    %automatically
    clear(pathGenParaFile);
    
    %generate parameter file for the test to run
    if PARAM_FILE_GEN_ON == 1     
        parameter_file_gen_json(dataFolder_test, dataFolder_calib, module_param_file, pathGenParaFile, dataPlatform);
    end
    
    %load calibration parameters
    load(dataFolder_calib)
    
    % simTopObj is used for top level parameter parsing and data loading and saving
    simTopObj           = simTopCascade('pfile', pathGenParaFile);
    calibrationObj      = calibrationCascade('pfile', pathGenParaFile, 'calibrationfilePath', dataFolder_calib);
    rangeFFTObj         = rangeProcCascade('pfile', pathGenParaFile);
    DopplerFFTObj       = DopplerProcClutterRemove('pfile', pathGenParaFile);
    detectionObj        = CFAR_CASO('pfile', pathGenParaFile);
    DOAObj              = DOACascade('pfile', pathGenParaFile);
    
    % get system level variables
    platform            = simTopObj.platform;
    numValidFrames      = simTopObj.totNumFrames;
    cnt = 1;
    frameCountGlobal = 0;
    
    
   % Get Unique File Idxs in the "dataFolder_test"   
   [fileIdx_unique] = getUniqueFileIdx(dataFolder_test);
    fprintf('fileIdx_unique = %3d\n', size(fileIdx_unique));
    for i_file = 1:(length(fileIdx_unique))
        
       % Get File Names for the Master, Slave1, Slave2, Slave3   
       [fileNameStruct]= getBinFileNames_withIdx(dataFolder_test, fileIdx_unique{i_file});        
       
      %pass the Data File to the calibration Object
      calibrationObj.binfilePath = fileNameStruct;
        
      detection_results = [];  
        
       % Get Valid Number of Frames 
       [numValidFrames dataFileSize] = getValidNumFrames(fullfile(dataFolder_test, fileNameStruct.masterIdxFile));
       %numValidFrames = 1;% number of valid frames is one less than what we
       %at capture scripts.
       fprintf('dataFilesize = %3d\n', dataFileSize);
       fprintf('number of valid frames= %3d\n', numValidFrames);
      
       fprintf('sample size= %3d bits \n', 16); 
       fprintf('number of samples per a chirp =  %3d\n', calibrationObj.numSamplePerChirp);
       fprintf('chirps per a loop = %3d \n',12);
       fprintf('loops per a frame = %3d\n',calibrationObj.nchirp_loops);
       fprintf('bits per a frame = %3d\n',calibrationObj.nchirp_loops*12*calibrationObj.numSamplePerChirp*16);
       bitsperframe = calibrationObj.nchirp_loops*12*calibrationObj.numSamplePerChirp*16;
       fprintf('number of valid frames= %3d\n', numValidFrames);
       fprintf('number of rx antennas=%3d\n', 4);
       fprintf('considering IQ= %3d\n', 2);
       fprintf('totalcapturedbits= %3d\n', bitsperframe*2*4*numValidFrames);
       fprintf('capturedsize in bytes= %3d\n', bitsperframe*2*4*numValidFrames/8);
       fprintf('capturedsize in KB= %3d KB\n', bitsperframe*2*4*numValidFrames/(8*1024));
       fprintf(['according to the lua script for mimo configuration, each of the 12 chirps ' ...
           'per a loop is transmitted in 12 tx antennas using time division multiplexing.' ...
           'This is a real mimo configuration.\n']);
       
       
        %intentionally skip the first frame due to TDA2 
        for frameIdx = 2:1:numValidFrames %numFrames_toRun [start_index:step: no.frames]
            tic
            %read and calibrate raw ADC data            
            calibrationObj.frameIdx = frameIdx;%
            frameCountGlobal = frameCountGlobal+1
            adcData = datapath(calibrationObj);%adc data contains the samples captured by all four devices. BUT for a one frame.
            disp(size(adcData)); %[256 128 16 12]: [SamplesPerChirp NumberOfLoopsPerFrame RX TX] 192 channels because of virtual antennas
            numElements = numel(adcData);
            disp(numElements);

            % RX Channel re-ordering
            adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:);            
            numElements = numel(adcData);
            disp(numElements);
            %only take TX and RXs required for MIMO data analysis
            % adcData = adcData
            outputDir = 'C:\Users\asus\Documents\Projects\FYP\DSP\Matlab\RadarSignalProcessingMatlab\4chip_cascade_MIMO_example';
            plotWaveletForChirps(adcData, outputDir, frameIdx);
            if mod(frameIdx, 10)==1
                fprintf('Processing %3d frame...\n', frameIdx);
            end
            
            
            
            %perform 2D FFT
            rangeFFTOut = [];
            DopplerFFTOut = [];
            
            for i_tx = 1: size(adcData,4)
                % range FFT
                rangeFFTOut(:,:,:,i_tx)     = datapath(rangeFFTObj, adcData(:,:,:,i_tx));
                
                % Doppler FFT
                DopplerFFTOut(:,:,:,i_tx)   = datapath(DopplerFFTObj, rangeFFTOut(:,:,:,i_tx));
                
            end
            disp(size(rangeFFTOut));
%            while(true)
%                pause(1);
%            end
            %plotWaveletForRangeBins(rangeFFTOut);
            % CFAR done along only TX and RX used in MIMO array
            DopplerFFTOut = reshape(DopplerFFTOut,size(DopplerFFTOut,1), size(DopplerFFTOut,2), size(DopplerFFTOut,3)*size(DopplerFFTOut,4));
            
            %detection
            sig_integrate = 10*log10(sum((abs(DopplerFFTOut)).^2,3) + 1);
                        
            detection_results = datapath(detectionObj, DopplerFFTOut);
            detection_results_all{cnt} =  detection_results;
            
            detect_all_points = [];
            for iobj = 1:length(detection_results)
                detect_all_points (iobj,1)=detection_results(iobj).rangeInd+1;
                detect_all_points (iobj,2)=detection_results(iobj).dopplerInd_org+1;
                detect_all_points (iobj,4)=detection_results(iobj).estSNR;
            end
            
            if PLOT_ON
                figure(1);
                set(gcf,'units','normalized','outerposition',[0 0 1 1])                
                subplot(2,2,1)               
                plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,size(sig_integrate,2)/2+1),'g','LineWidth',4);hold on; grid on
                for ii=1:size(sig_integrate,2)
                    plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,ii));hold on; grid on
                    if ~isempty(detection_results)
                        ind = find(detect_all_points(:,2)==ii);
                        if (~isempty(ind))
                            rangeInd = detect_all_points(ind,1);
                            plot(rangeInd*detectionObj.rangeBinSize, sig_integrate(rangeInd,ii),'o','LineWidth',2,...
                                'MarkerEdgeColor','k',...
                                'MarkerFaceColor',[.49 1 .63],...
                                'MarkerSize',6);
                        end
                    end
                end
                
                %title(['FrameID: ' num2str(cnt)]);
                xlabel('Range(m)');
                ylabel('Receive Power (dB)')
                title(['Range Profile(zero Doppler - thick green line): frameID ' num2str(frameIdx)]);
                hold off;
                subplot(2,2,2);
                %subplot_tight(2,2,2,0.1)
                imagesc((sig_integrate))
                c = colorbar;
                c.Label.String = 'Relative Power(dB)';
                title(' Range/Velocity Plot');
                pause(0.01)
            end
            
            angles_all_points = [];
            xyz = [];
            %if 0
            if ~isempty(detection_results)
                % DOA, the results include detection results + angle estimation results.
                % access data with angleEst{frame}(objectIdx).fieldName
                angleEst = datapath(DOAObj, detection_results);
                
                if length(angleEst) > 0
                    for iobj = 1:length(angleEst)
                        angles_all_points (iobj,1:2)=angleEst(iobj).angles(1:2);
                        angles_all_points (iobj,3)=angleEst(iobj).estSNR;
                        angles_all_points (iobj,4)=angleEst(iobj).rangeInd;
                        angles_all_points (iobj,5)=angleEst(iobj).doppler_corr;
                        angles_all_points (iobj,6)=angleEst(iobj).range;
                        %switch left and right, the azimuth angle is flipped
                        xyz(iobj,1) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                        xyz(iobj,2) = angles_all_points (iobj,6)*cosd(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                        %switch upside and down, the elevation angle is flipped
                        xyz(iobj,3) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,2)*-1);
                        xyz(iobj,4) = angleEst(iobj).doppler_corr;
                        xyz(iobj,9) = angleEst(iobj).dopplerInd_org;
                        xyz(iobj,5) = angleEst(iobj).range;
                        xyz(iobj,6) = angleEst(iobj).estSNR;
                        xyz(iobj,7) = angleEst(iobj).doppler_corr_overlap;
                        xyz(iobj,8) = angleEst(iobj).doppler_corr_FFT;
                        
                    end
                    angles_all_all{cnt} = angles_all_points;
                    xyz_all{cnt}  = xyz;
                    maxRangeShow = detectionObj.rangeBinSize*rangeFFTObj.rangeFFTSize;
                    %tic
                    if PLOT_ON
                        moveID = find(abs(xyz(:,4))>=0);
                        subplot(2,2,4);                        
                        
                        if cnt==1
                            scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                        else
                            yz = [xyz_all{cnt}; xyz_all{cnt-1}];
                            scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                        end
                        
                        c = colorbar;
                        c.Label.String = 'velocity (m/s)';                        
                        grid on;
                        
                        xlim([-20 20])
                        ylim([1 maxRangeShow])
                        %zlim([-4 4])
                        zlim([-5 5])
                        xlabel('X (m)')
                        ylabel('y (m)')
                        zlabel('Z (m)')                        
                        
                        view([-9 15])                        
                        title(' 3D point cloud');
                        
                        %plot range and azimuth heatmap
                        subplot(2,2,3)
                        STATIC_ONLY = 1;
                        minRangeBinKeep =  5;
                        rightRangeBinDiscard =  20;
                        [mag_data_static(:,:,frameCountGlobal) mag_data_dynamic(:,:,frameCountGlobal) y_axis x_axis]= plot_range_azimuth_2D(detectionObj.rangeBinSize, DopplerFFTOut,...
                            length(calibrationObj.IdTxForMIMOProcess),length(calibrationObj.RxForMIMOProcess), ...
                            detectionObj.antenna_azimuthonly, LOG_ON, STATIC_ONLY, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard);
                        title('range/azimuth heat map static objects')
                       
                        
    if (DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP)                   
    figure(2)
    subplot(121);
    surf(y_axis, x_axis, (mag_data_static(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
    view(2);
    xlabel('meters');    ylabel('meters')
    title({'Static Range-Azimuth Heatmap',strcat('Current Frame Number = ', num2str(frameCountGlobal))})
    
    subplot(122);
    surf(y_axis, x_axis, (mag_data_dynamic(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
    view(2);    
    xlabel('meters');    ylabel('meters')
    title('Dynamic HeatMap')
    end
    pause(0.1) 

     
                    end
                    
                end
                
            end
                             
            cnt = cnt + 1;    
       toc    
        end
        
        
    end
    
    ind = strfind(dataFolder_test, '\');
    testName = dataFolder_test(ind(end-1)+1:(ind(end)-1));
    if SAVEOUTPUT_ON == 1
        save(['.\main\cascade\output\newOutput_',testName,'.mat'],'angles_all_all', 'detection_results_all','xyz_all');
    end
    testID = testID + 1;
    
end
