
function rbm;

%using MNIST
%X_train, y_train = MNIST.traindata()
%X_train_rows = X_train';
%X_train_rows = X_train_rows[1:42000,:];
%X_train_cols = X_train[:,1:42000]

X_train = rand(5000,784);

X_batch_col = X_train(1:200,:)';
rbm = initializeRBM_col(784, 225, 0.01);
tic; contrastive_divergence_col_K(X_batch_col, rbm, 1, 0.01); toc;

X_batch_rows = X_train(1:200,:);
rbm = initializeRBM_rows(784,225, 0.01);
tic; contrastive_divergence_rows_K(X_batch_rows, rbm, 1, 0.01); toc;

return;

function x = sigmoid(v);
  x = 1./(1 + exp(-v));
  return;

function rbm = initializeRBM_col(n_vis, n_hid, sigma);
  rbm.W = sigma*randn(n_hid,n_vis);  % weight matrix
  rbm.vis_bias = zeros(n_vis,1);     % visible vector  
  rbm.hid_bias = zeros(n_hid,1);     % Hidden vector
  rbm.n_vis = n_vis;                 % num visible units 
  rbm.n_hid = n_hid;                 % num hidden unnits
  return;

function contrastive_divergence_col_K(Xbatch, rbm, K, lr);
        
    batch_size = size(Xbatch,2);

    Delta_W = zeros(size(rbm.W));
    Delta_b = zeros(size(rbm.vis_bias));
    Delta_c = zeros(size(rbm.hid_bias));

    for i = 1:batch_size
        x =  Xbatch(:,i);
        xneg = Xbatch(:,i);

        for k = 1:K
            hneg = sigmoid( rbm.W * xneg + rbm.hid_bias) > rand(size(rbm.n_hid));
            xneg = sigmoid( rbm.W' * hneg + rbm.vis_bias) > rand(size(rbm.n_vis));
        end

        ehp = sigmoid(rbm.W * x + rbm.hid_bias);
        ehn = sigmoid(rbm.W * xneg + rbm.hid_bias);
     
        Delta_W = Delta_W + lr * ( x * ehp' - xneg * ehn')';
        Delta_b = Delta_b + lr * (x - xneg);
        Delta_c = Delta_c + lr * (ehp - ehn);

    end

    rbm.W = rbm.W + Delta_W / batch_size;
    rbm.vis_bias = rbm.vis_bias + Delta_b / batch_size;
    rbm.hid_bias = rbm.hid_bias + Delta_c / batch_size;
    
    return 


function contrastive_divergence_rows_K(Xbatch, rbm, K, lr)
        
    batch_size = size(Xbatch,1);

    Delta_W = zeros(size(rbm.W));
    Delta_b = zeros(size(rbm.vis_bias));
    Delta_c = zeros(size(rbm.hid_bias));

    for i = 1:batch_size
        x =  Xbatch(i:i,:);
        xneg = Xbatch(i:i,:);

        for k = 1:K
            hneg = sigmoid( xneg * rbm.W + rbm.hid_bias') > rand(1,rbm.n_hid);
            xneg = sigmoid( hneg * rbm.W' + rbm.vis_bias') > rand(1,rbm.n_vis);
        end

        ehp = sigmoid(x * rbm.W + rbm.hid_bias');
        ehn = sigmoid(xneg * rbm.W + rbm.hid_bias');

        Delta_W = Delta_W + lr * ( ehp' * x - ehn' * xneg)';
        Delta_b = Delta_b + lr * (x - xneg)';
        Delta_c = Delta_c + lr * (ehp - ehn)';
    end
    
    rbm.W = rbm.W + Delta_W / batch_size;
    rbm.vis_bias = rbm.vis_bias + Delta_b / batch_size;
    rbm.hid_bias = rbm.hid_bias + Delta_c / batch_size;
    
    return 


function rbm = initializeRBM_rows(n_vis, n_hid, sigma)
  rbm.W = sigma*randn(n_vis,n_hid);  % weight matrix
  rbm.vis_bias = zeros(n_vis,1);     % visible vector  
  rbm.hid_bias = zeros(n_hid,1);     % Hidden vector
  rbm.n_vis = n_vis;                 % num visible units 
  rbm.n_hid = n_hid;                 % num hidden unnits
  return;

%%%%%%%%%%
