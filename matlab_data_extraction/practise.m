[numRows, numCols] = size(XTr);
chunkSize = 1;
fileID = fopen('./data/balanced_dataset_f.csv', 'w');
for col = 1:chunkSize:numRows
    f = XTr{col};
    f_t = f' ;
    n_f = f_t(:) ;
    final = n_f' ;
    writematrix(final, './data/balanced_dataset_f.csv', 'WriteMode', 'append');
    fprintf('Wrote columns %d to %d\n', col);
end 
fclose(fileID)