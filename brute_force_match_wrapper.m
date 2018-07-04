%% this is a matlab wrapper for the function that computes the distance between graphs via bruteforce
% we assume that we are going to read undirected, unweigted graphs of the same size

function [M] = brute_force_match_wrapper(A, B, pathtoprogram, cpugpu )

    rng shuffle;
    
    dim = size(A,1);
    
    fileA = ['graph', num2str(randi(1000000000)) ];
    fileB = ['graph', num2str(randi(1000000000)) ];
    fileoutput = ['outputmatch_', num2str(randi(1000000000)) ];   
    
    [row, column, val] = find(A);
    table = [row, column];
    dlmwrite(fileA, table, 'delimiter', ' ');
    
    [row, column, val] = find(B);
    table = [row, column];
    dlmwrite(fileB, table, 'delimiter', ' ');
    
    command_to_exec = [pathtoprogram , ' ',fileA ,' ', fileB , ' ', fileoutput,' 1 0 ', num2str(cpugpu),' ', num2str(dim)];
    system(command_to_exec);
    
    foutputID = fopen(fileoutput,'r');
    readoutput = fscanf(foutputID, '%d ',dim);
    fclose(foutputID);
    
    M = sparse(1:dim, readoutput , 1,dim,dim);
    
    delete(fileA);
    delete(fileB);
    delete(fileoutput);

end