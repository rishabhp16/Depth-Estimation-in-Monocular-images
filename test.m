pathtest = 'G:\IP Project\Project\testResize\'; % Path to resized test images
load('gist_features','gist');
fprintf('GIST features loaded\n');
RMSE = zeros(134,1);
RErr = zeros(134,1);
LTE = zeros(134,1);
NCC = zeros(134,1);
for k=1:1
        k
        filename=strcat(pathtest,sprintf('test%d.jpg',k));
        %fprintf('Test File=%s\n',filename);
        test_img=imread(filename);
        % Test image GIST
        clear param
        param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
        param.numberBlocks = 4;
        param.fc_prefilt = 4;
        fprintf('Computing gist...\n');
        gist_test = LMgist(test_img, '', param);
        D=struct('index','sim');
        for i=1:400
            D(i).index=gist(i).img;
            D(i).sim =sum((gist(i).val-gist_test).^2);
            %disp(D(i).img);
            %disp(D(i).sim);
        end;
        %bSORTIING STRUCTURE
        Dfields = fieldnames(D);
        Dcell = struct2cell(D);
        sz = size(Dcell);
        Dcell = reshape(Dcell, sz(1), []);   
        % Make each field a column
        Dcell = Dcell';                         %
        % Sort by first field "sim"
        Dcell = sortrows(Dcell, 2);
        % Put back into original cell array format
        Dcell = reshape(Dcell', sz);
        % Convert to Struct
        Dsorted = cell2struct(Dcell, Dfields, 1);

        %fprintf('Done extracting similiar images.\nFinding blur & color of Test image...\n');
        %Calculating blur values for test image
        test_av_blur=run_jnb(filename);
       
        test_av_blur= reshape(test_av_blur',[1 numel(test_av_blur)]);
        %Calculating color cues for test image
        color_=rgb2hsv(test_img);
        hue=color_(:,:,1);
        sat=color_(:,:,2);
        val=color_(:,:,3);

        %clear color_ D Dcell Dfields e gist gist

        hue_test = blockproc(hue,[16 16],@(x)mean2(x.data));
        hue_test = reshape((hue_test)',1,numel(hue_test)); 

        sat_test = blockproc(sat,[16 16],@(x)mean2(x.data));
        sat_test = reshape((sat_test)',1,numel(sat_test));

        val_test = blockproc(val,[16 16],@(x)mean2(x.data));
        val_test = reshape((val_test)',1,numel(val_test));

        test_color= [hue_test,sat_test , val_test];

        load('blur_input_ann.mat');
        load('target_depth.mat');
        load('color_cues.mat');
        train50_ip_blur = [];
        train50_tr = [];
        train50_ip_color=[];
        %Normalize test

%         hue_test = normalize(hue_test,hue_data);
%         sat_test = normalize(sat_test,sat_data);
%         val_test = normalize(val_test,val_data);
%         test_av_blur = normalize(test_av_blur,blur_input_ann);

        %Training ANN
%         hue_data_norm = normalize(hue_data,hue_data);
%         sat_data_norm = normalize(sat_data,sat_data);
%         val_data_norm = normalize(val_data,val_data);
%         blur_input_ann = normalize(blur_input_ann,blur_input_ann);

        % Picking best 50 images similiar to testing image
        for i=1:50
            idx = Dsorted(i).index;
       
            train50_ip_blur = [train50_ip_blur; blur_input_ann(idx,:)];
            temp= [hue_data(idx,:),sat_data(idx,:) , val_data(idx,:)];
            train50_ip_color = [train50_ip_color; temp];
            train50_tr = [train50_tr; target_depth(idx,:)];
            
        end
        
        fprintf('50 training images obtained\nBuilding BLUR ANN\n');
        blur_ANN_script
        fprintf('Building COLOR ANN\n');
        color_ANN_script
        
        
        % 3RD ANN
        train50_op_blur = [];
        train50_op_col = [];
        for i=1:50
            temp = blur_ANN(train50_ip_blur(i,:)');
            train50_op_blur = [train50_op_blur; temp'];
            temp = color_ANN(train50_ip_color(i,:)');
            train50_op_col = [train50_op_col ; temp' ];
        end
%         train50_op_blur_norm = normalize(train50_op_blur ,train50_op_blur );
%         train50_op_col_norm =  normalize(train50_op_col ,train50_op_col );
        train50_ip2 = [train50_op_blur , train50_op_col ];
        final_ANN_script 
        
        % Start test image
       
        result_blur = blur_ANN(test_av_blur'); % 638x1
        result_blur = result_blur' ;
        result_color= color_ANN([hue_test ,sat_test ,val_test]');  % 638x1
        result_color = result_color' ;
%         result_blur = normalize(result_blur,train50_op_blur);
%         result_color = normalize(result_color,train50_op_col);
        input_final = [result_blur  ,result_color  ];
        result_depth = final_ANN(input_final');
        est_depth = result_depth'; 
        %est_depth = (result_blur + result_color) / 2;
        load('test_depth');
        % result_blur  = 1x638
        % result_color = 1x638
        % est_depth    = 1x638   (avg)
        % test_depth   = 134x638
%         tt = reshape(est_depth,22,29);
%         tt = tt';
%         tt = medfilt2(tt);
%         est_depth = reshape(tt,1,[]);
        

        RMSE(k) = sqrt(sum((est_depth- test_depth(k,:)).^2)/638);      % generalize 1
        RErr(k) = sum(abs(est_depth-test_depth(k,:))./test_depth(k,:))/638;
        LTE(k) = sum(abs(log10(est_depth)-log10(test_depth(k,:))))/638;
        NCC(k) = sum((est_depth-mean(est_depth)).*(test_depth(k,:)-mean(test_depth(k,:)))) /(638*std(est_depth)*std(test_depth(k,:)));
end
%save('test_ANN_error_median','RMSE','RErr','LTE','NCC');