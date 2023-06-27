% Sample-level Multi-view Graph Clustering
function [res,Loss] = SLMVGC(data,labels, alpha, beta, lambda, gamma, k)
% data: cell array, view_num by 1, each array is num_samp by d_v
% num_clus: number of clusters
% num_view: number of views
% num_samp: number of instances
% labels: groundtruth of the data, num by 1
% lambda: trade-off parameter for ||Z^(v) - I||_F^2
% gamma : trade-off parameter for ||Z^(v) - Q^(v)||_F^2
% alpha: trade-off parameter for \sum_{i=1}^n ||S_i^T - W_i*(Z_hat)_i||_2^2
% beta: trade-off parameter for ||S||_F^2
% k: number of neighbors when initialize G

if nargin < 3
    alpha = 1;
end
if nargin < 4
    beta = 1;
end
if nargin < 5
    lambda = 1;
end
if nargin < 6
    gamma = 1;
end
if nargin < 7
    k = 10;
end
max_iter = 10;
%  ====== Initialization =======
num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels)); 
% initialize Gv
GV = cell(num_view,1);
sumG = zeros(num_samp);
for v = 1:num_view
    Gv = constructW_PKN(data{v}',k);
    GV{v} = Gv;
    sumG = sumG + Gv;
    clear Gv 
end 
% initialize S
S = sumG/num_view;
S0 = S-diag(diag(S));
w0 = (S0+S0')/2;
D0 = diag(sum(w0));
L = D0 - w0;
% initialize D
I = eye(num_samp);
% initialize W
W = ones(num_view, num_samp) / num_view;
% === initialize Zv Qv===
ZV = cell(num_view,1);
QV = cell(num_view,1);
for v = 1:num_view
    A = (lambda)*I + L;
    Q = QV{v};
    for i = 1:num_samp
        index = find(GV{v}(i,:)>0);
        Ii = I(i,index);
        b = 2*lambda*Ii; 
        % solve z^T*A*z-z^T*b
        [zi, ~] = fun_alm(A(index,index),b);
        ZV{v}(i,index) = zi';
    end
    Zv = ZV{v};
    Zv =Zv - diag(diag(Zv));
    ZV{v} = (Zv + Zv')/2;
    QV{v} = ZV{v};
end
% initialize Z_bar Q_bar
Z_bar = zeros(num_samp, num_samp, num_view);
Q_bar = zeros(num_samp, num_samp, num_view);
for i = 1:num_samp
    for v = 1:num_view
        Zv =ZV{v};
        Z_bar(:,i,v) = Zv(:,i);
        Q_bar(:,i,v) = Zv(:,i);
    end
end

% ================== iteration ==================
fprintf('begin updating ......\n')
iter = 0;
bstop = 0;
Loss = [];
% for iter = 1: Iter
while ~bstop
    iter = iter + 1;
    fprintf('the %d -th iteration ......\n', iter);
    % === update W ===
        for i = 1:num_samp
            Qi = zeros(num_view, num_samp);
            for v = 1:num_view
                Qi(v, :) = Q_bar(:, i, v)';
            end
            T = ones(num_view,1)*S(:,i)' - Qi;
            B = ((T*T')\ones(num_view, 1));
            C = ones(1, num_view)/(T*T')*ones(num_view, 1);
            W(:, i) = B / C;
        end
    %
    % === update Zv ===
    Z = cell(num_view,1);
    for v = 1:num_view
        A = (lambda + gamma)*I + L;
        Q = QV{v};
        for i = 1:num_samp
            index = find(GV{v}(i,:)>0);
            Ii = I(i,index);
            qi = Q(i,index);
            b = 2*lambda*Ii + 2*gamma*qi;
            % solve z^T*A*z-z^T*b
            [zi, ~] = fun_alm(A(index,index),b);
            ZV{v}(i,index) = zi';
        end
        Zv = ZV{v};
        Zv =Zv - diag(diag(Zv));
        ZV{v} = (Zv + Zv')/2;
    end
    %
    % === update Z_bar
    for v = 1:num_view
        Zv =ZV{v};
        for i = 1:num_samp
            Z_bar(:,i,v) = Zv(:,i);
        end
    end
    %
    % === update Q_bar
    for i = 1:num_samp
            Zi = zeros(num_view, num_samp);
            for v = 1:num_view
                Zi(v, :) = Z_bar(:, i, v)';
            end
            Qi = (gamma*eye(num_view) + alpha*W(:,i)*W(:,i)')\(gamma*Zi + alpha*W(:,i)*S(:,i)');
            Qi = max(Qi, 0);
            for v = 1:num_view
                Q_bar(:, i, v) = Qi(v, :)';
            end
    end
    %
    % === update Qv ===
    for v = 1:num_view
        for i = 1:num_samp
            Qv(:,i)  = Q_bar(:,i,v);
        end
        Qv = (Qv + Qv')/2;
        Qv = Qv - diag(diag(Qv));
        QV{v} = Qv;
    end
    %
    % === update S ===
    for i = 1:num_samp
        Qi = zeros(num_view, num_samp);
        for v = 1:num_view
            Qi(v, :) = Q_bar(:, i, v)';
        end
        h = -(-alpha*W(:,i)'*Qi)/(alpha+beta);
        S(i,:) = EProjSimplex_new(h);
    end
    S = (S + S')/2;
    
    % Calculate Loss
    L1_loss = 0; L2_loss = 0; L3_loss = 0; 
    for v=1:num_view
        L1_loss = L1_loss + trace(ZV{v}'*GV{v}*ZV{v}) + lambda*norm(ZV{v}-eye(num_samp),'fro')^2 + gamma*norm(ZV{v}-QV{v},'fro')^2;
    end
    for i = 1:num_samp
        Qi = zeros(num_view, num_samp);
        for v = 1:num_view
            Qi(v, :) = Q_bar(:, i, v)';
        end
        L2_loss = L2_loss + norm(S(:,i)' - W(:,i)'*Qi,'fro')^2;
    end
    L3_loss = L3_loss + norm(S,"fro")^2;
    Loss(iter) = L1_loss + alpha*L2_loss + beta*L3_loss;
    if (iter > 1) && ((iter > max_iter)||(abs(Loss(iter-1)-Loss(iter))/Loss(iter-1) <= 1e-6))
        bstop = 1;
    end
end

y = SpectralClustering(S, num_clus);
res = EvaluationMetrics(labels, y);
end

function [v, obj] = fun_alm(A,b)
if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
mu = 30;
n = size(A,1);
lambda = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/mu+lambda/mu;

    % update v
    c = A*z-b;
    d = lambda/mu-z;
    mm = d+c/mu;
    v = EProjSimplex_new(-mm);

    % update lambda
    lambda = lambda+mu*(v-z);
    mu = rho*mu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end

function [x] = EProjSimplex_new(v, k)
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end


