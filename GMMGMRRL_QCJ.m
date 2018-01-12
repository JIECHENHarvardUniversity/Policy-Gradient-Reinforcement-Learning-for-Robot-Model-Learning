%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This file contains a sample implementation of PoWER for the TSM manipulator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The algorithm is implemented with a number of simplifications:
% - the variance of the exploration is constant over trials
% - the exploration is constant during the trial
%   (as the motor primitives employ basis functions that are localized in time 
%   which are only active for a short period of time,
%   time-varying exploration does not have large effetcs)
% - the return is used instead of the state-action value function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
% Initilization of the EM Tracker
tracker_setup; 
% Open the serial port
global s;
s = serial('COM5');
set(s,'BaudRate',57600);
fopen(s);
fwrite(s,strcat('BP0',num2str(1000)));
pause(10);
% Calibrate the the center of the target with EM Tracker
% The unit is mm
x0=13.63*25.4;
y0=5.97*25.4;
z0=-5.86*25.4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of the number of components used in GMM.
nbStates = 16;
% Load a dataset consisting of six demonstrations of the motor commands
load('motor_1031.mat');
Data=motor_1031;
nbVar = size(Data,1);
% Number of samples within each demonstration
% The data has been preprocessed by DTW
nbSamples=1225;
GMRSamples=17;
% Variance of the exploration for EM based Reinforcement Learning
variance = 0.01.*ones((nbVar-1)*nbStates,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training of GMM by EM algorithm, initialized by K-means clustering.
[Priors, Mu, Sigma] = EM_init_kmeans(Data, nbStates);
[Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);
% Number of iterations for the EM based Reinforcement Learning
n_iter = 20;
% Define the reward function and parameters
Return = zeros(1,n_iter+1);
Q = zeros(GMRSamples,n_iter+1);
s_Return = zeros(n_iter+1,2);
param = zeros((nbVar-1)*nbStates,n_iter+1);
% Initialize parameters for the EM based Reinforcement Learning
param(:,1) = reshape(Mu(2:nbVar,:),(nbVar-1)*nbStates,1); 
current_param = [Mu(1,:);reshape(param(:,1),(nbVar-1),nbStates)];
% Use of GMR to retrieve a generalized version of the data 
% Generalize motor commands for the TSM manipulator
expData(1,:) = linspace(min(Data(1,:)), max(Data(1,:)), GMRSamples);
[expData(2:nbVar,:), expSigma] = GMR(Priors, current_param, Sigma, expData(1,:), [1], [2:nbVar]);
generalized_rm=expData(2,:);
generalized_lm=expData(3,:);
generalized_zb=expData(4,:);
generalized_rm(generalized_rm>1)=0.5;
generalized_rm(generalized_rm<(-1))=-0.5;
generalized_lm(generalized_lm>1)=0.5;
generalized_lm(generalized_lm<(-1))=-0.5;
generalized_zb(generalized_zb>1)=0.25;
generalized_zb(generalized_zb<(-1))=-0.25;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute the motor commands and receive end-effector positions from EM tracker
ZB=0.0220*5000+5000;
OCR4A=-0.2206*200+1500;
if(ZB<1000)
    fwrite(s,strcat('BP0',num2str(ZB)));
else
    fwrite(s,strcat('BP',num2str(ZB)));
end
pause(10);
fwrite(s,strcat('RP',num2str(OCR4A)));
pause(0.02);
[EndPos]=Move_motor(generalized_rm,generalized_lm,generalized_zb);
x=EndPos(:,1);
y=EndPos(:,2);
z=EndPos(:,3);
%theta=linspace(0,2*pi,GMRSamples);
REFY=[y0+30,y0+30,y0+30,y0+15,y0,y0-15,y0-30,y0-30,y0-30,y0-30,y0-30,y0-15,y0,y0+15,y0+30,y0+30,y0+30];
%REFZ=30*sin(theta)+z0;
REFZ=[z0,z0+15,z0+30,z0+30,z0+30,z0+30,z0+30,z0+15,z0,z0-15,z0-30,z0-30,z0-30,z0-30,z0-30,z0-15,z0];
REFX=x0*ones(1,GMRSamples);
plot3(REFX,REFY,REFZ);
axis equal;
hold on;
for i=1:GMRSamples
    plot3(x(i),y(i),z(i),'ro','MarkerSize',4);
    hold on;
    pause(0.025);
end
traj=[x,y,z];

for i=1:GMRSamples
    Q(1:(end+1-i),1) = Q(1:(end+1-i),1) + exp(-(abs(sqrt((z(i)-REFZ(i))^2+(y(i)-REFY(i))^2))+abs(x(i)-REFX(i))));
end
Q(:,1) = Q(:,1)./GMRSamples;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Conduct the iterations for the EM based Reinforcement Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:n_iter
    Return(iter) = Q(1,iter);
    disp(['Iteration=', num2str(iter)]);
    disp(['Return=', num2str(Return(iter))]);
    disp(['*----------------------------*']);
    
    % This lookup table will be used for the importance sampling,
    % The largest reture will be moved to the end of the s_Reture mattrix
    s_Return(1,:) = [Return(iter) iter];
    s_Return = sortrows(s_Return);
    
    % Update the policy parameters
    param_nom = zeros((nbVar-1)*nbStates,1);
    param_dnom = zeros((nbVar-1)*nbStates,1);  
    
    % Calculate the expectations (the normalization is taken care of by the division)
    % As importance sampling we take the 3 best rollouts
    for i=1:min(iter,3)
        % Get the rollout number for the 3 best rollouts
        j = s_Return(end+1-i,2);
		% calculate the exploration with respect to the current parameters
        % if you have time-varying exploration use 
        % temp_explore = (reshape(param(:,:,j),length(rt),n_rfs)-ones(length(rt),1)*current_param')';
        % instead
        temp_explore = (ones(GMRSamples,1)*(param(:,j)-reshape(current_param(2:nbVar,:),(nbVar-1)*nbStates,1))')';
		% repeat the Q values
        temp_Q = (Q(:,j)*ones(1,(nbVar-1)*nbStates))';
        param_nom = param_nom + sum(temp_explore.*temp_Q,2);
        param_dnom = param_dnom + sum(temp_Q,2);
    end
    
    % Update the parameters
    param(:,iter+1) = reshape(current_param(2:nbVar,:),(nbVar-1)*nbStates,1) + param_nom./(param_dnom+1.e-10);
    % Set the new mean of the parameters
    current_param = [Mu(1,:);reshape(param(:,iter+1),nbVar-1,nbStates)];
    
    % In the last rollout we want to get the return without exploration
    if iter~=n_iter
        param(:,iter+1) = param(:,iter+1) + variance.^.5.*randn(nbVar*nbStates,1);
    end
        % Apply the new parameters to the motor primitve
    %% Use of GMR to retrieve a generalized version of the data and associated
    %% constraints. A sequence of temporal values is used as input, and the 
    %% expected distribution is retrieved. 
    expData(1,:) = linspace(min(Data(1,:)), max(Data(1,:)), GMRSamples);
    [expData(2:nbVar,:), expSigma] = GMR(Priors, current_param, Sigma, expData(1,:), [1], [2:nbVar]);
    %% Generalize motor commands for the TSM manipulator
    generalized_rm=expData(2,:);
    generalized_lm=expData(3,:);
    generalized_zb=expData(4,:);
    generalized_rm(generalized_rm>1)=0.5;
    generalized_rm(generalized_rm<(-1))=-0.5;
    generalized_lm(generalized_lm>1)=0.5;
    generalized_lm(generalized_lm<(-1))=-0.5;
    generalized_zb(generalized_zb>1)=0.25;
    generalized_zb(generalized_zb<(-1))=-0.25;
    %% Execute the motor commands and receive end-effector trajectory from EM tracker
    [EndPos]=Move_motor(generalized_rm,generalized_lm,generalized_zb);
    x=EndPos(:,1);
    y=EndPos(:,2);
    z=EndPos(:,3);
    
    for i=1:GMRSamples
        plot3(x(i),y(i),z(i),'ro','MarkerSize',4);
        hold on;
        pause(0.025);
    end
    traj=[traj,x,y,z];
    % run the motor primitive
    for i=1:GMRSamples
		% calculate the Q values
        Q(1:(end+1-i),iter+1) = Q(1:(end+1-i),iter+1) + exp(-(abs(sqrt((z(i)-REFZ(i))^2+(y(i)-REFY(i))^2))+abs(x(i)-REFX(i))));
    end
	% normalize the Q values
    Q(:,iter+1) = Q(:,iter+1)./GMRSamples;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate the return of the final rollout
Return(iter+1) = Q(1,iter+1);
% plot the return over the rollouts
figure(1);
plot(Return);
ylabel('Return');
xlabel('Rollouts');
disp(['Final Return ', num2str(Return(end))]);
tracker_close;
fclose(s);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%