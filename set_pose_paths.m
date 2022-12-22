function [parentdir,jsondir,pyenvpath,rpath,binpath,codepath,ephyspath] = set_pose_paths(addCode)
% [parentdir,jsondir,pyenvpath,rpath,binpath,codepath,ephyspath] = set_pose_paths(addCode)
%
% adjust paths here to fit your local machine


% figure out the computer
c = computer;
if strcmp(c,'PCWIN64')
    u = getenv('USERNAME');
else
    [~,h] = system('hostname');
    h(end)=[];
    u = getenv('USER');
end

if 0
    parentdir = 'D:\';
    jsondir = 'xxx';
    pyenvpath='xxx';
    rpath = 'Rscript';
    binpath = '/usr/bin';
    codepath = 'C:\Users\HaydenLab\Documents\git\ana_pose_neural_embed';
    ephyspath = [codepath '/docs'];
else

    % paths
    if strcmp(c,'MACI64') && strcmpi(u,'ben') %BV
        parentdir = '/Volumes/SSD_Q';
        jsondir = '/Volumes/SSD_Q/Data_json_annot';
        pyenvpath = '/Users/ben/embed/bin/python3';
        rpath = '/usr/local/bin/Rscript';
        binpath = '/usr/local/bin';
        codepath = '/Users/Ben/Documents/git/oms_internal';
    elseif strcmp(c,'GLNXA64') && strcmp(h,'freyr') %freyr
        parentdir = '/mnt/scratch/BV_embed';
        jsondir = '/mnt/scratch/BV_embed/Data_json_annot';
        pyenvpath='/home/auser/miniconda3/envs/bv/bin/python';
        rpath = 'Rscript';
        binpath = '/usr/bin';
        %codepath = '/mnt/scratch/git/oms_internal';
        codepath = '/mnt/scratch/git/ana_pose_neural_embed';
        ephyspath = '/mnt/scratch/git/oms_internal/ephys/docs';
    elseif strcmp(c,'GLNXA64') && strcmp(h,'vidar') %vidar
        parentdir = '/mnt/scratch3/BV_embed';
        jsondir = '/mnt/scratch3/BV_embed/Data_json_annot';
        pyenvpath='/home/zlab_recon/miniconda3/envs/bv/bin/python';
        rpath = 'Rscript';
        binpath = '/usr/bin';
        codepath = '/mnt/scratch3/git/oms_internal';
        ephyspath = '/mnt/scratch3/git/oms_internal/ephys/docs';
    elseif strcmp(c,'GLNXA64') && strcmp(h,'loki') %loki
        parentdir = '/mnt/scratch/BV_embed';
        jsondir = '/mnt/scratch/BV_embed/Data_json_new';
        pyenvpath = '/home/zlab_recon/miniconda3/envs/bv/bin/python3';
        rpath = '';
        binpath = '/usr/bin';
        codepath = '/mnt/scratch/git/oms_internal';
        ephyspath = '/mnt/scratch/git/oms_internal/ephys/docs';
    else
        error('cant figure out the comp!')
    end
end
%}
