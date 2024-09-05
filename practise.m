[numRows, numCols] = size(XTr);
chunkSize = 1;
fileID = fopen('./data/XTr1.csv', 'w');
for col = 1:chunkSize:numCols
    f = XTr{col};
    f_t = f' ;
    n_f = f_t(:) ;
    final = n_f' ;
    writematrix(final, './data/XTr1.csv', 'WriteMode', 'append');
    fprintf('Wrote columns %d to %d\n', col);
end 
fclose(fileID)