%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
% Load the training data set
load('C:\Users\Administrator\Desktop\IL+RL\LFD_IK_FK_RL\traj.mat');
load('C:\Users\Administrator\Desktop\IL+RL\LFD_IK_FK_RL\mot.mat');
load('C:\Users\Administrator\Desktop\IL+RL\LFD_IK_FK_RL\x0.mat');
load('C:\Users\Administrator\Desktop\IL+RL\LFD_IK_FK_RL\y0.mat');
load('C:\Users\Administrator\Desktop\IL+RL\LFD_IK_FK_RL\z0.mat');
rm=mot(1,:);
lm=mot(2,:);
zb=mot(3,:);
x=traj(1,:)-x0;
y=traj(2,:)-y0;
z=traj(3,:)-z0;
% Apply polynomial regression and normal equation to learn the inverse
% kinematic model of the TSM manipulator
N=length(x);
train_f=zeros(N,4);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Use the proposed learning IK to track a circle located at (z0,y0),with radius 30mm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=0:0.01:2*pi;
M=length(t);
cx=zeros(1,M);
cy=30*cos(t);
cz=30*sin(t);
circle_f=zeros(M,4);
for i=1:M
    %circle_f(i,:)=[1,cx(i),cy(i),cz(i),cx(i)^2,cy(i)^2,cz(i)^2,cx(i)*cy(i),cx(i)*cz(i),cy(i)*cz(i),cx(i)^3,cy(i)^3,cz(i)^3,cx(i)*cy(i)*cz(i)];
    circle_f(i,:)=[1,cx(i),cy(i),cz(i)];
end
c_rm=circle_f*theta_rm;
c_lm=circle_f*theta_lm;
c_zb=circle_f*theta_zb;

% Use FK to calculate the end-effector trajectory
c_f=[c_rm,c_lm,c_zb]/([theta_rm,theta_lm,theta_zb]);

plot3(circle_f(:,2),circle_f(:,3),circle_f(:,4),'r-');
axis equal;
hold on;
plot3(c_f(:,2),c_f(:,3),c_f(:,4),'bo');
axis equal;



