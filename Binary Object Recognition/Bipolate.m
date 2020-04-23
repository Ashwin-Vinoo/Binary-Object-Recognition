function output = Bipolate(img,y,x)
    output=(img(y-1,x)+img(y,x+1)+img(y+1,x)+img(y,x-1)+0.7*(img(y-1,x-1)+img(y-1,x+1)+img(y+1,x+1)+img(y+1,x-1)))/6.8;
end