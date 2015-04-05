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
	B = [B log(str2num(lines{2}))];
end
fclose(f);
imgs={}; %red color for test
for i = 1:size(file_lists,2)
	img = imread(file_lists{i});
	imgs{end+1}=img;
end
sample_num = 1000;
sample_w = randi([1 size(img,1)],sample_num,1);
sample_h = randi([1 size(img,2)],sample_num,1);
Z = zeros(sample_num,size(file_lists,2));
g={};
lE={};
for color = 1:3
	for i = 1:size(file_lists,2)
		for j = 1:sample_num
			Z(j,i) = imgs{i}(sample_w(j),sample_h(j),color);
		end
	end

	[g{end+1},lE{end+1}]=gsolve(Z,B,l,w);

end
