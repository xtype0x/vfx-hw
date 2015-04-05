%read file info
%disp('VFX running~~');
%fflush(stdout);
f = fopen('list.txt','r');
file_lists={};
B=[];w=zeros(1,256);
l=100;
%initialize w
for i = 1:size(w,2)
	w(i) = i-1;
	if (i-1) > 127
		w(i) = 256 -i;
	end
end
%load list.txt
while feof(f)==0
	lines = strsplit(fgetl(f));
	file_lists{end+1} = lines{1};
	B = [B log(str2num(lines{2}))];
end
fclose(f);
%{
%load imgs
%disp('Loading images...');
%fflush(stdout);
imgs={};
for i = 1:size(file_lists,2)
	img = imread(file_lists{i});
	imgs{end+1}=img;
end
file_num = size(file_lists,2);

%sampling and radiance map recovering
%disp('Sampling and radiance mapping...');
%fflush(stdout);
sample_num = 500;
sample_w = randi([1 size(img,1)],sample_num,1);
sample_h = randi([1 size(img,2)],sample_num,1);
Z = zeros(sample_num,size(file_lists,2),3);
g={};
for k = 1:3
	for i = 1:file_num
		for j = 1:sample_num
			Z(j,i,k) = imgs{i}(sample_w(j),sample_h(j),k);
		end
	end
end
[g{1},lE_r]=gsolve(Z(:,:,1),B,l,w);
[g{2},lE_g]=gsolve(Z(:,:,2),B,l,w);
[g{3},lE_b]=gsolve(Z(:,:,3),B,l,w);

%output hdr pic
%fucking slow...

% disp('Output hdr file...');
% fflush(stdout);
% hdr = zeros(size(img,1),size(img,2),4);
% progress=0;
% for i = 1:size(img,1)
% 	for j = 1:size(img,2)
% 		%set rgb mid and E
% 		total_w=0;total_exposure=0;
% 		for c = 1:3
% 			max_val=0;min_val=256;
% 			for k = 1:file_num
% 				max_val = max(imgs{k}(i,j,c),max_val);
% 				min_val = min(imgs{k}(i,j,c),min_val);
% 				total_exposure += w(1+imgs{k}(i,j,c))*(g{c}(imgs{k}(i,j,c)+1)-B(k));
% 				total_w += w(imgs{k}(i,j,c)+1);
% 			end
% 			hdr(i,j,c) = round((max_val+min_val)/2);
% 		end
% 		hdr(i,j,4) = round(exp(total_exposure/total_w));
% 		progress++;
% 		disp(sprintf('%.2f%%',round(progress*10000/size(img,1)/size(img,2))/100));
% 		fflush(stdout);
% 	end
% end

%disp('output hdr...');
%fflush(stdout);
progress=0;
output_num=1;
choose = round(size(imgs,2)/2);
img = imgs{1};
img = cat(3,img,zeros(size(img,1),size(img,2)));
for i = 1:size(img,1)
	for j = 1:size(img,2)
		total_w=0;
		total_exposure=0;
		for c = 1:3
			total_exposure = total_exposure + w(1+img(i,j,c))*(g{c}(img(i,j,c)+1)-B(1));
			total_w = total_w + w(img(i,j,c)+1);
		end
		if total_w == 0
			img(i,j,4) = round(exp(total_exposure/3))
		else
			img(i,j,4)= round(exp(total_exposure/total_w));
		end
		progress=progress+1;
		%if progress/size(img,1)/size(img,2) >= output_num
			%disp(sprintf('%d%% complete',output_num));
            %output_num=output_num+1;
			%fflush(stdout);
		%end
	end
end

%}
img=makehdr(file_lists,'ExposureValues',floor(exp(B)));
hdrwrite(img,'output.hdr');
RGB=tonemap(img);
figure; imshow(RGB);
