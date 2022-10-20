function [parentdir,jsondir,pyenvpath,rpath] = set_pose_paths(addCode)
% [parentdir,jsondir,pyenvpath,rpath] = set_pose_paths(addCode)
%
% adjust paths here to fit your local machine

% add toolboxes that cant live on github
if addCode
%     s = '/Users/ben/Downloads/mat_bigcage';
    s = '/Users/david/Desktop/mat_bigcage';
    addpath(genpath(s))
end

% figure out the computer
c = computer;
if strcmp(c,'PCWIN64')
    u = getenv('USERNAME');
else
    u = getenv('USER');
end

% paths
if strcmp(c,'MACI64') && strcmpi(u,'ben') %BV
    %parentdir = '/Volumes/DATA_bg/ana'; 
    %jsondir = '/Volumes/DATA_bg/Data_json_new';
    parentdir = '/Volumes/SSD_Q/P_embedding';
    jsondir = '/Volumes/SSD_Q/P_embedding/Data_json_annot';
    pyenvpath = '/Users/ben/embed/bin/python3';
    rpath = '/usr/local/bin/Rscript';
elseif strcmp(c,'GLNXA64') && strcmp(u,'auser') %freyr
    parentdir = '/mnt/scratch/BV_embed';
    %jsondir = '/mnt/scratch/BV_embed/Data_all';
    %jsondir = '/mnt/scratch/BV_embed/Data_json_new';
    jsondir = '/mnt/scratch/BV_embed/Data_json_annot';
    pyenvpath='/home/auser/miniconda3/bin/python';
    rpath = 'Rscript';
elseif strcmp(c,'GLNXA64') && strcmp(u,'zlab_recon') %loki
    parentdir = '/mnt/scratch/BV_embed';
    jsondir = '/mnt/scratch/BV_embed/Data_json_new';
    pyenvpath = '';
    rpath = '';
else
    error('cant figure out the comp!')
end

