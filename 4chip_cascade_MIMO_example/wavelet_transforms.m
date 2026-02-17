% Define file path and parameters
fileFullPath = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\normal_environment\slave1_0000_data.bin';
frameIdx = 1;              % Index of the frame to read
numSamplePerChirp = 256;    % Number of samples per chirp
numChirpPerLoop = 12;       % Number of chirps per loop
numLoops = 128;              % Number of loops per frame
numRXPerDevice = 4;         % Number of receiving channels per device
numDevices = 4;             % Number of devices in the cascade (if needed)
sample_rate = 8e6;          % Sampling rate (in Hz)

% Read radar data
adcDataComplex = readBinFile(fileFullPath, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice);
disp(adcDataComplex);

% Select antenna index and extract chirp ADC matrix
antennaIdx = 1;  
chirp_ADC_matrix = adcDataComplex(:, :, antennaIdx, :);

% Display matrix size (should be [numSamplePerChirp, numLoops, numChirpsPerLoop])
disp(size(chirp_ADC_matrix));

% Extract first chirp of the first loop
first_chirp_first_loop = chirp_ADC_matrix(:, 1, 1); % This array consists of 256 complex ADC samples
disp(size(first_chirp_first_loop));  % Should be [numSamplePerChirp, 1]
% Define the file path where the complex values will be saved
filePath = 'first_chirp_first_loop.txt';

% Open the file in write mode
fileID = fopen(filePath, 'w');

% Write each complex value to the file line by line
for i = 1:length(first_chirp_first_loop)
    fprintf(fileID, '%f + %fi\n', real(first_chirp_first_loop(i)), imag(first_chirp_first_loop(i)));
end

% Close the file
fclose(fileID);

disp(['File written to: ', filePath]);

% Define wavelet and widths for CWT
wavelet = 'cmor2.5-1.0';  % Complex Morlet wavelet
widths = 1:64;  % Scale range for CWT, adjust based on signal resolution

% Define time-sampled array (in seconds) based on sampling rate
time_sampled = (0:numSamplePerChirp-1) / sample_rate; 

% Apply CWT on the extracted chirp data
%[cwtmatr, freqs_new] = cwt(first_chirp_first_loop, widths, wavelet, 'SamplingPeriod', 1/sample_rate);
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

% Function to read the binary radar data file
function [adcData1Complex] = readBinFile(fileFullPath, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice)
    Expected_Num_SamplesPerFrame = numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * 2;
    fp = fopen(fileFullPath, 'r');
    
    if fp == -1
        error('File could not be opened.');
    end
    
    % Move to the desired frame in the file
    fseek(fp, (frameIdx - 1) * Expected_Num_SamplesPerFrame * 2, 'bof');
    adcData1 = fread(fp, Expected_Num_SamplesPerFrame, 'uint16');
    fclose(fp);
    
    % Convert the 16-bit data to signed integers
    neg = logical(bitget(adcData1, 16));
    adcData1(neg) = adcData1(neg) - 2^16;
    
    % Combine the I and Q channels into complex values
    adcData1 = adcData1(1:2:end) + 1j * adcData1(2:2:end);
    
    % Reshape and permute the data
    adcData1Complex = reshape(adcData1, numRXPerDevice, numSamplePerChirp, numChirpPerLoop, numLoops);
    adcData1Complex = permute(adcData1Complex, [2 4 1 3]);
end
