%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This file contains a sample implementation of PoWER for the TSM manipulator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
x0=12.25*25.4;
y0=6.78*25.4;
z0=-5.94*25.4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the training data set
load('mot.mat');
load('traj.mat');
rm=mot(1,:);
lm=mot(2,:);
zb=mot(3,:);
x=traj(1,:)-337.5660;
y=traj(2,:)-183.8960;
z=traj(3,:)-(-151.1300);
% Apply polynomial regression and normal equation to learn the inverse
% kinematic model of the TSM manipulator
nbFeatures=4;
N=length(x);
train_f=zeros(N,nbFeatures);
for i=1:N
    %train_f(i,:)=[1,x(i),y(i),z(i),x(i)^2,y(i)^2,z(i)^2,x(i)*y(i),x(i)*z(i),y(i)*z(i),x(i)^3,y(i)^3,z(i)^3,x(i)*y(i)*z(i)];
    train_f(i,:)=[1,x(i),y(i),z(i)];
end
theta_rm=pinv(train_f'*train_f)*train_f'*rm';
theta_lm=pinv(train_f'*train_f)*train_f'*lm';
theta_zb=pinv(train_f'*train_f)*train_f'*zb';
% Calculate the predicated motor commands
Test_rm=train_f*theta_rm;
Test_lm=train_f*theta_lm;
Test_zb=train_f*theta_zb;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the prediction errors
ERM=sum((Test_rm-rm').*(Test_rm-rm'))/N;
ELM=sum((Test_lm-lm').*(Test_lm-lm'))/N;
EZB=sum((Test_zb-zb').*(Test_zb-zb'))/N;
% Calculate the average square error of the regression algorithm
error=ERM+ELM+EZB;
disp(['The Average Square Error of the Polynomial Regression Algorithm=', num2str(error)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the exploration variance for the RL algorithm
variance = 0.05.*ones(nbFeatures,1);
% Number of iterations
n_iter = 20;
% Define the rewards and parameters
Return = zeros(1,n_iter+1);
s_Return = zeros(n_iter+1,2);
param_rm = zeros(nbFeatures,n_iter+1);
param_lm = zeros(nbFeatures,n_iter+1);
param_zb = zeros(nbFeatures,n_iter+1);

% Initialize parameters for the EM-RL
param_rm(:,1) = theta_rm; 
param_lm(:,1) = theta_lm;
param_zb(:,1) = theta_zb;
current_rm = param_rm(:,1);
current_lm = param_lm(:,1);
current_zb = param_zb(:,1);
%% Generalize motor commands for the TSM manipulator
% Define the reference end-effector trajectory
t=0:0.1:2*pi;
M=length(t);
cx=zeros(1,M);
cy=30*cos(t);
cz=30*sin(t);
circle_f=zeros(M,nbFeatures);
for i=1:M
    %circle_f(i,:)=[1,cx(i),cy(i),cz(i),cx(i)^2,cy(i)^2,cz(i)^2,cx(i)*cy(i),cx(i)*cz(i),cy(i)*cz(i),cx(i)^3,cy(i)^3,cz(i)^3,cx(i)*cy(i)*cz(i)];
    circle_f(i,:)=[1,cx(i),cy(i),cz(i)];
end
c_rm = circle_f*current_rm;
c_lm = circle_f*current_lm;
c_zb = circle_f*current_zb;
% Plot the results
c_f=[c_rm,c_lm,c_zb]/([current_rm,current_lm,current_zb]);
plot3(circle_f(:,2),circle_f(:,3),circle_f(:,4),'r-');
hold on;

c_rm(c_rm>0.25)=0.25;
c_rm(c_rm<(-0.25))=-0.25;
c_lm(c_lm>0.25)=0.25;
c_lm(c_lm<(-0.25))=-0.25;
c_zb(c_zb>0.15)=0.15;
c_zb(c_zb<(-0.15))=-0.15;
%% Execute the motor commands and receive end-effector trajectory from EM tracker
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
pause(1);
[EndPos]=Move_motor(c_rm,c_lm,c_zb);
xx=EndPos(:,1);
yy=EndPos(:,2);
zz=EndPos(:,3);
for i=1:M
    plot3(xx(i)-x0,yy(i)-y0,zz(i)-z0,'bo','MarkerSize',4);
    hold on;
    pause(0.025);
end
Q = zeros(M,n_iter+1);
for i=1:M
    Q(1:(end+1-i),1) = Q(1:(end+1-i),1) + exp(-(abs(sqrt((zz(i)-z0)^2+(yy(i)-y0)^2))+2*abs(xx(i)-x0)));
end
Q(:,1) = Q(:,1)./M;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do the iterations
for iter=1:n_iter
    if (mod(iter,5)==0)
        disp(['Iter ', num2str(iter)]);
    end
    
    Return(iter) = Q(1,iter);
    disp(['Iteration=', num2str(iter)]);
    disp(['Return=', num2str(Return(iter))]);
    % this lookup table will be used for the importance sampling,
    % the largest reture will be moved to the end of the s_Reture mattrix
    s_Return(1,:) = [Return(iter) iter];
    s_Return = sortrows(s_Return);
    
    % update the policy parameters
    param_nom_rm = zeros(nbFeatures,1);
    param_nom_lm = zeros(nbFeatures,1);
    param_nom_zb = zeros(nbFeatures,1);
    param_dnom = zeros(nbFeatures,1); 
    
    % calculate the expectations (the normalization is taken care of by the division)
    % as importance sampling we take the 2 best rollouts
    for i=1:min(iter,2)
        % get the rollout number for the 2 best rollouts
        j = s_Return(end+1-i,2);
        % calculate the exploration with respect to the current parameters        
        temp_explore_rm = (ones(M,1)*(param_rm(:,j)-current_rm)')';
        temp_explore_lm = (ones(M,1)*(param_lm(:,j)-current_lm)')';
        temp_explore_zb = (ones(M,1)*(param_zb(:,j)-current_zb)')';
        temp_Q = (Q(:,j)*ones(1,nbFeatures))';
        % as we use the return, always have the same exploration variance,
        % and assume that always only one basis functions is active we get 
        % these simple sums
        param_nom_rm = param_nom_rm + sum(temp_explore_rm.*temp_Q,2);
        param_nom_lm = param_nom_lm + sum(temp_explore_lm.*temp_Q,2);
        param_nom_zb = param_nom_zb + sum(temp_explore_zb.*temp_Q,2);
        param_dnom = param_dnom + sum(temp_Q,2);
    end
    
    % update the parameters
    param_rm(:,iter+1) = current_rm + param_nom_rm./(param_dnom+1.e-10);
    param_lm(:,iter+1) = current_lm + param_nom_lm./(param_dnom+1.e-10);
    param_zb(:,iter+1) = current_zb + param_nom_zb./(param_dnom+1.e-10);
        
    % in the last rollout we want to get the return without exploration
    if iter~=n_iter
        param_rm(:,iter+1) = param_rm(:,iter+1) + variance.^.5.*(2*rand(nbFeatures,1)-1);
        param_lm(:,iter+1) = param_lm(:,iter+1) + variance.^.5.*(2*rand(nbFeatures,1)-1);
        param_zb(:,iter+1) = param_zb(:,iter+1) + variance.^.5.*(2*rand(nbFeatures,1)-1);
    end
    
    % set the new mean of the parameters
    current_rm = param_rm(:,iter+1);
    current_lm = param_lm(:,iter+1);
    current_zb = param_zb(:,iter+1);
    
    % Generalize motor commands
    c_rm = circle_f*current_rm;
    c_lm = circle_f*current_lm;
    c_zb = circle_f*current_zb;
    c_rm(c_rm>0.25)=0.25;
    c_rm(c_rm<(-0.25))=-0.25;
    c_lm(c_lm>0.25)=0.25;
    c_lm(c_lm<(-0.25))=-0.25;
    c_zb(c_zb>0.15)=0.15;
    c_zb(c_zb<(-0.15))=-0.15;
    %% Execute the motor commands and receive end-effector trajectory from EM tracker
    [EndPos]=Move_motor(c_rm,c_lm,c_zb);
    xx=EndPos(:,1);
    yy=EndPos(:,2);
    zz=EndPos(:,3);
    for i=1:M
        plot3(xx(i)-x0,yy(i)-y0,zz(i)-z0,'bo','MarkerSize',4);
        hold on;
        pause(0.025);
    end
    pause(2);
    for i=1:M
		% calculate the Q values
        Q(1:(end+1-i),1) = Q(1:(end+1-i),1) + exp(-(abs(sqrt((zz(i)-z0)^2+(yy(i)-y0)^2))+2*abs(xx(i)-x0)));
    end
	% normalize the Q values
    Q(:,iter+1) = Q(:,iter+1)./M;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate the return of the final rollout
sum=0;
for i=1:M
    sum=sum+abs(sqrt((zz(i)-z0)^2+(yy(i)-y0)^2)-30)+abs(xx(i)-x0);
end
Return(iter+1)=exp(-sum/M);

% plot the return over the rollouts
figure(1);
plot(Return);
ylabel('Return');
xlabel('Rollouts');

disp(['Final Return ', num2str(Return(end))]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%