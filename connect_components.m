function D = connect_components(D,Q)

    blocks = components(D)';
    total_numblocks=max(blocks);
    
    iters=1;
    while(iters<total_numblocks)
        blocks = components(D)';
        count = zeros(1, max(blocks));
        cnn_comps={};
        numblocks=max(blocks);
      
        for i=1:numblocks
            count(i) = length(find(blocks == i));
            cnn_comps{i}=find(blocks == i);
        end
    
        tmp_dis=zeros(numblocks,numblocks);%block之间的距离
        for i=1:numblocks
            for j=1:numblocks
                t=Q(cnn_comps{i},cnn_comps{j});         
                tmp_dis(i,j)=min(min(t));
            end
            tmp_dis(i,i)=inf;
        end
    
        [conn_i,conn_j]=find(tmp_dis==min(min(tmp_dis)));%block之间最近
        if length(conn_i)>1
            conn_j=conn_i(2);
            conn_i=conn_i(1);
        end       
        dis_conn=Q(cnn_comps{conn_i},cnn_comps{conn_j});
        [row,column]=find(dis_conn==min(min(dis_conn)));
        i=cnn_comps{conn_i}(row);
        j=cnn_comps{conn_j}(column);
        D(i,j)=min(min(dis_conn));
        D(j,i)=min(min(dis_conn));
        iters=iters+1;
    end
    
    


    