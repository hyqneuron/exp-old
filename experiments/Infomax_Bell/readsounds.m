
% READSOUNDS looks in a directory "sounds/" for sound files  and 
%   returns them in a NxP matrix called "sounds" where N is the number 
%   of sounds specified, and P is the length of the shortest one (the 
%   others are truncated). One caveat: the filenames MUST all have the 
%   same number of characters since they are stored in a matrix (what else?!).
%
%   Example call: 
%         sounds=readsounds(['word2';'word1']);

function sounds=readsounds(files)
  minlen=1e10;
  for fileno=1:size(files,1),
    fprintf('reading %s \n', files(fileno,:));
    temp=auread(['/home/tony/Matlab/sounds/' files(fileno,:)])';
    len=size(temp,2);
    if minlen>len, minlen=len; end;
    sounds(fileno,1:minlen)=temp(1:minlen);
  end;
  sounds=sounds(:,1:minlen);
