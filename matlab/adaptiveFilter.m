%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The seed
rng(42);

% 10 second files I have prepared
%[s, Fs] = audioread('PF_1.wav');
%[s, Fs] = audioread('taunt.wav');
Fs = 22257;

% Ten seconds of sample
N = Fs * 10;
%N = length(s);

% Generate speach like noise between 85-255Hz
% white noise -> sharp BPF
w_speach_L = 85*2*pi;
%w_speach_H = 255*2*pi;
%w_speach_H = 2550*2*pi;
w_speach_H = 22000*2*pi;
white_noise = randn(N, 1);
theta_speach_L = w_speach_L / (2 * pi * Fs);
theta_speach_H = w_speach_H / (2 * pi * Fs);
[B_LPF, A_LPF] = butter(6, theta_speach_H, 'low');
[B_HPF, A_HPF] = butter(6, theta_speach_L, 'high');
s = filter(B_LPF, A_LPF, white_noise);
s = filter(B_HPF, A_HPF, s);

% Generate sinusoidal interference
w_0 = 28160*2*pi;
%A_noise = 0.02864;  % ~+10dB SNR for faux-speach input 255Hz
%A_noise = 0.13864;  % ~+10dB SNR for faux-speach input 2550Hz
A_noise = 0.45;  % ~+10dB SNR for faux-speach input 22000Hz
%A_noise = 0.099;    % ~+10dB SNR for taunt.wav
%A_noise = 1.0;         % ~-10dB for taunt.wav
theta_0 = w_0 / (2 * pi * Fs);
v = zeros(N, 1);
for n = 1:N
    d_theta = 0;                         % No deviation
    %d_theta = 0.02 * cos(0.00015 * pi * n);  % Predictable deviation (fast)
    %d_theta =  0.02 * cos(0.00003 * pi * n);  % Predictable deviation (slow)
    %d_theta = normrnd(0,1);             % Random deviation
    v(n) = A_noise * cos((theta_0 + d_theta) * n);
end

% Calculate SNR
Ps = var(s);
Pv = var(v);
snr = 10 * log10(Ps/Pv)

% add noise to the signal
x_recieved = s + v;

% plot the sample_audio
fft_w = fft(s);
P2 = abs(fft_w/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(N/2))/N;
%plot(f,P1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make a[] an array so we can 
% see how it changed over time
a = zeros(N, 1);

% Initialize output vec
y = zeros(N, 1);
y(1) = x_recieved(1);
y(2) = x_recieved(2);

% Run the filter in a real time simulation
r = 0.99;
%mu = 0.075; % voice
%mu = 0.75; % noise 255Hz
%mu = 0.075; % noise 2550Hz
mu = 0.00075; % noise 2550Hz
k = (1+r)/2;
a_true = 2*cos(theta_0)
for n = 3:N
    
    y(n) = k*x_recieved(n) - k*a(n)*x_recieved(n-1) + k*x_recieved(n-2) + r*a(n)*y(n-1) - r*r*y(n-2);
    %y(n) = k*x_recieved(n) - k*a_true*x_recieved(n-1) + k*x_recieved(n-2) + r*a_true*y(n-1) - r*r*y(n-2);
    
    % Update n
    a_n1 = a(n) + mu*y(n)*(x_recieved(n-1) - r*y(n-1));
    if (a_n1 < -2 || 2 < a_n1)
        a_n1 = 0;
    end
    a(n+1) = a_n1;
end

% display a
figure();
plot(a);

% display e2
e = y - s;
e2 = e.^2;
figure();
plot(e2);

%y = filter([k -k*a_true k], [1 -r*a_true r*r], x_recieved);

% play the sound before trans -> after noise -> after filter
%sound(s, Fs);
%pause(4)
%sound(x_recieved, Fs);
%pause(4)
%sound(y, Fs);
