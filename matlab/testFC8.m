caffe.set_mode_cpu()
caffe.reset_all()

avg = cat(3,122.67891434,116.66876762,104.00698793);

net = caffe.Net('/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascal-deploy.prototxt','/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascal.caffemodel','test');

I=imread('/home/gipadmin/Downloads/people.jpg');

subplot(121)
I = imresize(I,[500 500]); I_=I;
imshow(I_)

I = single(I) - repmat(avg,[500 500]);
I = I(:,:,[3 2 1]);
I = permute(I,[2 1 3]);

net.blobs('data').set_data(reshape(I,[500 500 3 1]));

net.forward_prefilled();

[m,ind]=max(net.blobs('upscore').get_data(),[],3);

subplot(122)

imagesc(ind')
axis equal;
axis tight;
axis off;