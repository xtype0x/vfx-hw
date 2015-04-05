%read file info
f = fopen('list.txt','r');
file_lists={};
B=[];w=zeros(1,256);
l=50;
for i = 1:size(w,2)
	w(i) = i-1;
	if (i-1) > 127
		w(i) = 256 -i;
	end
end

while feof(f)==0
	lines = strsplit(fgetl(f));
	file_lists{end+1} = lines{1};
	B = [B 1 / str2num(lines{2})];
end
fclose(f);

%hdr = makehdr(file_lists,'RelativeExposure', B ./ B(1));

% Z={}; %red color for test
% for i = 1:size(file_lists,2)
% 	img = imread(file_lists{i});
% 	Z{end+1}=img;
% end

%[g,lE]=gsolve(Z,B,l,w);

