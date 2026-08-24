function [ IDX ] = local_knn( coor,nlink,nremove )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   This function is used to generate network by local knnsearch and MST
%   Inpur parameters:
%   coor            : coordinate vector
%   nlink           : Number of arcs that link to one start point
%   Output parameters:
%   IDX.from        : index of start point
%   IDX.to          : index of end point
%   Written by LIANG Hongyu @ PolyU-LSGI, June-29-2018
% -------------------------------------------------------------------------
%% check input
narginchk(1,3);
if nargin < 3
    nremove = 5;
end
if nargin < 2
    nlink = 20;
end
nremove=max(nremove,0);
NPS=size(coor,1);
%% local knn
Pcls=knnsearch(coor,coor,'k',nlink+nremove);
Pcls(:,2:1+nremove)=[];
sp=(repmat(Pcls(:,1),1,nlink-1))';   
from = reshape(sp,(nlink-1)*NPS,1); 
to   = reshape(Pcls(:,2:end)',(nlink-1)*NPS,1); 
Arctem1=[from,to];
%% Minimal spanning tree to link each sub-network
DT=DelaunayTri(coor);
% E = edges(DT);
% nEdges=size(E,1);
% Dx2=(coor(E(:,1),1)-coor(E(:,2),1)).^2;
% Dy2=(coor(E(:,1),2)-coor(E(:,2),2)).^2;
% dis =sqrt(Dx2+Dy2);
% dis2=sparse(E(:,1),E(:,2),dis,NPS,NPS,nEdges);
% D2=dis2+dis2';
% D2=tril(D2);
% G2=graphminspantree(D2);
% [ii,jj]=find(G2~=0);
% Arctem2=[ii,jj];
Arctem2=edges(DT);


Arc=[Arctem1;Arctem2];
Arc = unique(sort(Arc,2),'rows');

% remove long arc
% dis=sqrt((coor(Arc(:,1),1)-coor(Arc(:,2),1)).^2+(coor(Arc(:,1),2)-coor(Arc(:,2),2)).^2);
% idx=dis<50;
% Arc=Arc(idx,:);


IDX.to=Arc(:,2);
IDX.from=Arc(:,1);
%EOF
end

