%**************************PATTERN RECOGNITION USING HISTOGRAM SIMILARITY IDENTIFICATION*******************************% 
clc;
clear all;
close all;
%% OBTAINING THE IMAGE TO BE PROCESSED FROM AN IMAGE FILE OR CAMERA
i=0;
BW_thresh=50;
while(i~=1&&i~=2&&i~=3)
    i=input('To obtain the image through the camera, Enter 1. To open a JPEG image image, Enter 2:');
    if(i==1)    
        vid=videoinput('winvideo',1,'YUY2_320x240');  
        while (i==1) 
            preview(vid);                                         
            disp('Enter any key to take a snapshot');
            pause;                                                                  
            img=getsnapshot(vid);                                                                       
            disp('Snapshot image has been succesfully obtained, Please view the processed image');
            closepreview(vid);                                     
            img=single(rgb2gray(ycbcr2rgb(img)));
            j=min(img(:))+BW_thresh;
            for x=1:320
                for y=1:240
                    if(img(y,x)>=j)
                        img(y,x)=1;
                    else
                        img(y,x)=0;
                    end;
                end;
            end;
            imtool(img,[0,1]);                                            
            i=input('To repeat photoshoot,Enter 1. To continue Enter 2:');    
            imtool close all;
        end;
        delete(vid);
    elseif(i==2)   
        [x,y]=uigetfile('*.jpg','Open The Required JPG image'); 
        if(strfind(x,'.jpg'))
            img=rgb2gray(imread(strcat(y,x),'jpg'));
            img=single(imresize(img,[240,320]));
            j=min(img(:))+BW_thresh;
            for x=1:320
                for y=1:240
                    if(img(y,x)>=j)
                        img(y,x)=1;
                    else
                        img(y,x)=0;
                    end;
                end;
            end;
        else 
            disp('JPG Image aquisition error');
            i=0;
        end;
    else
        disp('An Error has Occured in Entering the choice, Please repeat');
    end;
end;
%% ALGORITHM'S I, II, III, IV & V- LOCAL BINARY PATTERN, ROBUST LOCAL BINARY PATTERN, DISCRIMINATIVE ROBUST LOCAL BINARY PATTERN, LOCAL TERNARY PATTERN, ROBUST LOCAL TERNARY PATTERN 
LTP_Threshold=0.3;
LBP_bin=zeros(1,57);
RLBP_bin=zeros(1,29);
ULBP_bin=zeros(1,57);
LLBP_bin=zeros(1,57);
RLTP_bin=zeros(1,6561);
HLBP_bin=zeros(1,256);
DRLBP_bin=zeros(1,58);
HLTP_bin=zeros(1,6561);
DRLTP_bin=zeros(1,6561);
for x=3:318
    for y=3:238
        LBP(y,x)=128*(Bipolate(img,y-1,x-1)-img(y,x)>=0)+64*(Bipolate(img,y-1,x)-img(y,x)>=0)+32*(Bipolate(img,y-1,x+1)-img(y,x)>=0)+16*(Bipolate(img,y,x+1)-img(y,x)>=0)+8*(Bipolate(img,y+1,x+1)-img(y,x)>=0)+4*(Bipolate(img,y+1,x)-img(y,x)>=0)+2*(Bipolate(img,y+1,x-1)-img(y,x)>=0)+(Bipolate(img,y,x-1)-img(y,x)>=0)+1;
        i=is_uniform(LBP(y,x));
        if(i)
            LBP_bin(i)=LBP_bin(i)+1; 
            i=is_uniform(min(LBP(y,x),257-LBP(y,x)));
            RLBP_bin(i)=RLBP_bin(i)+1; 
        elseif(LBP(y,x)~=256 && LBP(y,x)~=1);
            LBP_bin(57)=LBP_bin(57)+1;
            RLBP_bin(29)=RLBP_bin(29)+1; 
        end;
        if(LBP(y,x)~=1&&LBP(y,x)~=256)
            weight(y,x)=sqrt((-0.5*Bipolate(img,y,x-1)+0.5*Bipolate(img,y,x+1)).^2+(-0.5*Bipolate(img,y-1,x)+0.5*Bipolate(img,y+1,x)).^2);
            HLBP_bin(LBP(y,x))=HLBP_bin(LBP(y,x))+weight(y,x);
        end;
        ULBP(y,x)=128*(Bipolate(img,y-1,x-1)-img(y,x)>=LTP_Threshold)+64*(Bipolate(img,y-1,x)-img(y,x)>=LTP_Threshold)+32*(Bipolate(img,y-1,x+1)-img(y,x)>=LTP_Threshold)+16*(Bipolate(img,y,x+1)-img(y,x)>=LTP_Threshold)+8*(Bipolate(img,y+1,x+1)-img(y,x)>=LTP_Threshold)+4*(Bipolate(img,y+1,x)-img(y,x)>=LTP_Threshold)+2*(Bipolate(img,y+1,x-1)-img(y,x)>=LTP_Threshold)+(Bipolate(img,y,x-1)-img(y,x)>=LTP_Threshold)+1;
        i=is_uniform(ULBP(y,x));
        if(i)
            ULBP_bin(i)=ULBP_bin(i)+1; 
        elseif(ULBP(y,x)~=256 && ULBP(y,x)~=1);
            ULBP_bin(57)=ULBP_bin(57)+1;
        end;  
        LLBP(y,x)=128*(Bipolate(img,y-1,x-1)-img(y,x)<=-LTP_Threshold)+64*(Bipolate(img,y-1,x)-img(y,x)<=-LTP_Threshold)+32*(Bipolate(img,y-1,x+1)-img(y,x)<=-LTP_Threshold)+16*(Bipolate(img,y,x+1)-img(y,x)<=-LTP_Threshold)+8*(Bipolate(img,y+1,x+1)-img(y,x)<=-LTP_Threshold)+4*(Bipolate(img,y+1,x)-img(y,x)<=-LTP_Threshold)+2*(Bipolate(img,y+1,x-1)-img(y,x)<=-LTP_Threshold)+(Bipolate(img,y,x-1)-img(y,x)<=-LTP_Threshold)+1;                
        i=is_uniform(LLBP(y,x));
        if(i)
            LLBP_bin(i)=LLBP_bin(i)+1; 
        elseif(LLBP(y,x)~=256 && LLBP(y,x)~=1);
            LLBP_bin(57)=LLBP_bin(57)+1;
        end; 
        LTP(y,x,1:8)=[(bitget(ULBP(y,x)-1,8)-bitget(LLBP(y,x)-1,8)),(bitget(ULBP(y,x)-1,7)-bitget(LLBP(y,x)-1,7)),(bitget(ULBP(y,x)-1,6)-bitget(LLBP(y,x)-1,6)),(bitget(ULBP(y,x)-1,5)-bitget(LLBP(y,x)-1,5)),(bitget(ULBP(y,x)-1,4)-bitget(LLBP(y,x)-1,4)),(bitget(ULBP(y,x)-1,3)-bitget(LLBP(y,x)-1,3)),(bitget(ULBP(y,x)-1,2)-bitget(LLBP(y,x)-1,2)),(bitget(ULBP(y,x)-1,1)-bitget(LLBP(y,x)-1,1))];
        i=(LTP(y,x,1)+1)*3.^7+(LTP(y,x,2)+1)*3.^6+(LTP(y,x,3)+1)*3.^5+(LTP(y,x,4)+1)*3.^4+(LTP(y,x,5)+1)*3.^3+(LTP(y,x,6)+1)*3.^2+(LTP(y,x,7)+1)*3.^1+(LTP(y,x,8)+2);       
        HLTP_bin(i)=HLTP_bin(i)+1;
        j=(-LTP(y,x,1)+1)*3.^7+(-LTP(y,x,2)+1)*3.^6+(-LTP(y,x,3)+1)*3.^5+(-LTP(y,x,4)+1)*3.^4+(-LTP(y,x,5)+1)*3.^3+(-LTP(y,x,6)+1)*3.^2+(-LTP(y,x,7)+1)*3.^1+(-LTP(y,x,8)+2);       
        Invert_value(i)=j;
        j=3281;
        for i=1:8
            if(LTP(y,x,i)==1)
                j=(LTP(y,x,1)+1)*3.^7+(LTP(y,x,2)+1)*3.^6+(LTP(y,x,3)+1)*3.^5+(LTP(y,x,4)+1)*3.^4+(LTP(y,x,5)+1)*3.^3+(LTP(y,x,6)+1)*3.^2+(LTP(y,x,7)+1)*3.^1+(LTP(y,x,8)+2);
                break;
            elseif(LTP(y,x,i)==-1)
                j=(-LTP(y,x,1)+1)*3.^7+(-LTP(y,x,2)+1)*3.^6+(-LTP(y,x,3)+1)*3.^5+(-LTP(y,x,4)+1)*3.^4+(-LTP(y,x,5)+1)*3.^3+(-LTP(y,x,6)+1)*3.^2+(-LTP(y,x,7)+1)*3.^1+(-LTP(y,x,8)+2);
                break;                
            end;
        end;
        if(j~=3281)
            RLTP_bin(j)=RLTP_bin(j)+1;
        end;         
    end;
end;
for i=1:128
    j=is_uniform(i);
    if(j~=0)
        DRLBP_bin(j)=HLBP_bin(i)+HLBP_bin(257-i);
        DRLBP_bin(j+29)=abs(HLBP_bin(i)-HLBP_bin(257-i));
    else
        DRLBP_bin(29)=DRLBP_bin(29)+HLBP_bin(i)+HLBP_bin(257-i);
        DRLBP_bin(58)=DRLBP_bin(58)+abs(HLBP_bin(i)-HLBP_bin(257-i));
    end;
end;
for i=1:3280
    if(HLTP_bin(i)~=0)
        if(i~=1)
            DRLTP_bin(i)=HLTP_bin(i)+HLTP_bin(Invert_value(i));
        else
            DRLTP_bin(1)=HLTP_bin(1);
        end;
        DRLTP_bin(i+3280)=abs(HLTP_bin(i)-HLTP_bin(Invert_value(i)));
    end;
end;
LBP_bin(:)=LBP_bin(:)/sum(LBP_bin);
RLBP_bin(:)=RLBP_bin(:)/sum(RLBP_bin);
DRLBP_bin(:)=DRLBP_bin(:)/sum(DRLBP_bin);
LTP_bin=horzcat(ULBP_bin,LLBP_bin);
LTP_bin(:)=LTP_bin(:)/sum(LTP_bin);
RLTP_bin(:)=RLTP_bin(:)/sum(RLTP_bin);
DRLTP_bin(1:3280)=DRLTP_bin(1:3280)/(2*sum(DRLTP_bin(1:3280)));
DRLTP_bin(3281:6561)=DRLTP_bin(3281:6561)/(2*sum(DRLTP_bin(3281:6561)));
%% DISPLAY THE LBP, RLBP, LTP & RLTP ALGORITHM'S BINS AS 4 HISTOGRAMS
figure;
subplot(3,2,1);
bar(LBP_bin);
title('LBP HISTOGRAM');
xlabel('LBP bins');
ylabel('Occurence Probability');
subplot(3,2,3);
bar(RLBP_bin);
title('RLBP HISTOGRAM');
xlabel('RLBP bins');
ylabel('Occurence Probability');
subplot(3,2,5);
bar(DRLBP_bin);
title('DRLBP HISTOGRAM');
xlabel('DRLBP bins');
ylabel('Occurence Probability');
subplot(3,2,2);
bar(LTP_bin,'r');
title('LTP HISTOGRAM');
xlabel('LTP bins');
ylabel('Occurence Probability');
subplot(3,2,4);
bar(RLTP_bin,'r');
title('RLTP HISTOGRAM');
xlabel('RLTP bins');
ylabel('Occurence Probability');
subplot(3,2,6);
bar(DRLTP_bin,'r');
title('DRLTP HISTOGRAM');
xlabel('DRLTP bins');
ylabel('Occurence Probability');
%% IDENTIFICATION OF FIGURE BY COMPARING CURRENT HISTOGRAMS WITH STANDARD STORED HISTOGRAMS OF SEVERAL SHAPES
clc;
load('Square_Standard_bins.mat');
load('Rectangle_Standard_bins.mat');
load('Triangle_Standard_bins.mat');
load('Circle_Standard_bins.mat');
load('Pentagon_Standard_bins.mat');
load('Diamond_Standard_bins.mat');
Square_LBP_Similarity=(2-sum(abs(Square_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Rectangle_LBP_Similarity=(2-sum(abs(Rectangle_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Triangle_LBP_Similarity=(2-sum(abs(Triangle_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Circle_LBP_Similarity=(2-sum(abs(Circle_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Pentagon_LBP_Similarity=(2-sum(abs(Pentagon_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Diamond_LBP_Similarity=(2-sum(abs(Diamond_Standard_LBP_bin(:)-LBP_bin(:))))/2;
Sim_Max=max([Square_LBP_Similarity,Rectangle_LBP_Similarity,Triangle_LBP_Similarity,Circle_LBP_Similarity,Pentagon_LBP_Similarity,Diamond_LBP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_LBP_Similarity)
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_LBP_Similarity*100));
    elseif(Sim_Max==Rectangle_LBP_Similarity)
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_LBP_Similarity*100));     
    elseif(Sim_Max==Triangle_LBP_Similarity)
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_LBP_Similarity*100));  
    elseif(Sim_Max==Circle_LBP_Similarity)
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_LBP_Similarity*100));  
    elseif(Sim_Max==Pentagon_LBP_Similarity)
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_LBP_Similarity*100));  
    else
        fprintf('LBP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_LBP_Similarity*100));          
    end;
else
    disp('LBP algorithm fails to identify the current figure');
end;
Square_RLBP_Similarity=(2-sum(abs(Square_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Rectangle_RLBP_Similarity=(2-sum(abs(Rectangle_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Triangle_RLBP_Similarity=(2-sum(abs(Triangle_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Circle_RLBP_Similarity=(2-sum(abs(Circle_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Pentagon_RLBP_Similarity=(2-sum(abs(Pentagon_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Diamond_RLBP_Similarity=(2-sum(abs(Diamond_Standard_RLBP_bin(:)-RLBP_bin(:))))/2;
Sim_Max=max([Square_RLBP_Similarity,Rectangle_RLBP_Similarity,Triangle_RLBP_Similarity,Circle_RLBP_Similarity,Pentagon_RLBP_Similarity,Diamond_RLBP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_RLBP_Similarity)
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_RLBP_Similarity*100));
    elseif(Sim_Max==Rectangle_RLBP_Similarity)
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_RLBP_Similarity*100));     
    elseif(Sim_Max==Triangle_RLBP_Similarity)
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_RLBP_Similarity*100));  
    elseif(Sim_Max==Circle_RLBP_Similarity)
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_RLBP_Similarity*100));  
    elseif(Sim_Max==Pentagon_RLBP_Similarity)
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_RLBP_Similarity*100));  
    else
        fprintf('RLBP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_RLBP_Similarity*100));          
    end;
else
    disp('RLBP algorithm fails to identify the current figure');
end;
Square_DRLBP_Similarity=(2-sum(abs(Square_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Rectangle_DRLBP_Similarity=(2-sum(abs(Rectangle_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Triangle_DRLBP_Similarity=(2-sum(abs(Triangle_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Circle_DRLBP_Similarity=(2-sum(abs(Circle_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Pentagon_DRLBP_Similarity=(2-sum(abs(Pentagon_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Diamond_DRLBP_Similarity=(2-sum(abs(Diamond_Standard_DRLBP_bin(:)-DRLBP_bin(:))))/2;
Sim_Max=max([Square_DRLBP_Similarity,Rectangle_DRLBP_Similarity,Triangle_DRLBP_Similarity,Circle_DRLBP_Similarity,Pentagon_DRLBP_Similarity,Diamond_DRLBP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_DRLBP_Similarity)
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_DRLBP_Similarity*100));
    elseif(Sim_Max==Rectangle_DRLBP_Similarity)
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_DRLBP_Similarity*100));     
    elseif(Sim_Max==Triangle_DRLBP_Similarity)
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_DRLBP_Similarity*100));  
    elseif(Sim_Max==Circle_DRLBP_Similarity)
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_DRLBP_Similarity*100));  
    elseif(Sim_Max==Pentagon_DRLBP_Similarity)
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_DRLBP_Similarity*100));  
    else
        fprintf('DRLBP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_DRLBP_Similarity*100));          
    end;
else
    disp('DRLBP algorithm fails to identify the current figure');
end;
Square_LTP_Similarity=(2-sum(abs(Square_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Rectangle_LTP_Similarity=(2-sum(abs(Rectangle_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Triangle_LTP_Similarity=(2-sum(abs(Triangle_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Circle_LTP_Similarity=(2-sum(abs(Circle_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Pentagon_LTP_Similarity=(2-sum(abs(Pentagon_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Diamond_LTP_Similarity=(2-sum(abs(Diamond_Standard_LTP_bin(:)-LTP_bin(:))))/2;
Sim_Max=max([Square_LTP_Similarity,Rectangle_LTP_Similarity,Triangle_LTP_Similarity,Circle_LTP_Similarity,Pentagon_LTP_Similarity,Diamond_LTP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_LTP_Similarity)
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_LTP_Similarity*100));
    elseif(Sim_Max==Rectangle_LTP_Similarity)
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_LTP_Similarity*100));     
    elseif(Sim_Max==Triangle_LTP_Similarity)
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_LTP_Similarity*100));  
    elseif(Sim_Max==Circle_LTP_Similarity)
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_LTP_Similarity*100));  
    elseif(Sim_Max==Pentagon_LTP_Similarity)
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_LTP_Similarity*100));  
    else
        fprintf('LTP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_LTP_Similarity*100));
    end;
else
    disp('LTP algorithm fails to identify the current figure');
end;
Square_RLTP_Similarity=(2-sum(abs(Square_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Rectangle_RLTP_Similarity=(2-sum(abs(Rectangle_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Triangle_RLTP_Similarity=(2-sum(abs(Triangle_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Circle_RLTP_Similarity=(2-sum(abs(Circle_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Pentagon_RLTP_Similarity=(2-sum(abs(Pentagon_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Diamond_RLTP_Similarity=(2-sum(abs(Diamond_Standard_RLTP_bin(:)-RLTP_bin(:))))/2;
Sim_Max=max([Square_RLTP_Similarity,Rectangle_RLTP_Similarity,Triangle_RLTP_Similarity,Circle_RLTP_Similarity,Pentagon_RLTP_Similarity,Diamond_RLTP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_RLTP_Similarity)
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_RLTP_Similarity*100));
    elseif(Sim_Max==Rectangle_RLTP_Similarity)
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_RLTP_Similarity*100));     
    elseif(Sim_Max==Triangle_RLTP_Similarity)
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_RLTP_Similarity*100));  
    elseif(Sim_Max==Circle_RLTP_Similarity)
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_RLTP_Similarity*100));  
    elseif(Sim_Max==Pentagon_RLTP_Similarity)
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_RLTP_Similarity*100));  
    else
        fprintf('RLTP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_RLTP_Similarity*100));          
    end;
else
    disp('RLTP algorithm fails to identify the current figure');
end;
Square_DRLTP_Similarity=(2-sum(abs(Square_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Rectangle_DRLTP_Similarity=(2-sum(abs(Rectangle_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Triangle_DRLTP_Similarity=(2-sum(abs(Triangle_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Circle_DRLTP_Similarity=(2-sum(abs(Circle_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Pentagon_DRLTP_Similarity=(2-sum(abs(Pentagon_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Diamond_DRLTP_Similarity=(2-sum(abs(Diamond_Standard_DRLTP_bin(:)-DRLTP_bin(:))))/2;
Sim_Max=max([Square_DRLTP_Similarity,Rectangle_DRLTP_Similarity,Triangle_DRLTP_Similarity,Circle_DRLTP_Similarity,Pentagon_DRLTP_Similarity,Diamond_DRLTP_Similarity]);
if(Sim_Max>0.66)
    if(Sim_Max==Square_DRLTP_Similarity)
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Square with percentage similarity:%d o/o\n',uint8(Square_DRLTP_Similarity*100));
    elseif(Sim_Max==Rectangle_DRLTP_Similarity)
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Rectangle with percentage similarity:%d o/o\n',uint8(Rectangle_DRLTP_Similarity*100));     
    elseif(Sim_Max==Triangle_DRLTP_Similarity)
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Triangle with percentage similarity:%d o/o\n',uint8(Triangle_DRLTP_Similarity*100));  
    elseif(Sim_Max==Circle_DRLTP_Similarity)
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Circle with percentage similarity:%d o/o\n',uint8(Circle_DRLTP_Similarity*100));  
    elseif(Sim_Max==Pentagon_DRLTP_Similarity)
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Pentagon with percentage similarity:%d o/o\n',uint8(Pentagon_DRLTP_Similarity*100));  
    else
        fprintf('DRLTP Histogram comparison shows that the Pattern obtained is that of a Diamond with percentage similarity:%d o/o\n',uint8(Diamond_DRLTP_Similarity*100));          
    end;
else
    disp('DRLTP algorithm fails to identify the current figure');
end;