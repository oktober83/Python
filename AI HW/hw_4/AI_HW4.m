%% AI HW4 - Problem 3
close all, clear all, clc
format long
N = [100; 200; 500; 1000];
x = [];
psi = @(x) 0.5 + sign(x)/2 * (sqrt(1 - exp(-2*(x^2)/pi)));
upperLimit =  psi(5);
lowerLimit = psi(-5);
buckets = linspace(-5,5,51);
bucketValues = zeros(51,1);
bucketAmounts = zeros(50,1);
for i = 1:51
    bucketValues(i) = psi(buckets(i));
end
for i = 1:50
    bucketAmounts(i) = (buckets(i) + buckets(i+1)) / 2;
end

x1 = zeros(N(1),1);
for j = 1:N(1);
    r = rand;
    while r > upperLimit || r < lowerLimit
        r = rand;
    end
    i = 2;
    while r > bucketValues(i)
        i = i + 1;
    end
    i
    x1(j) = bucketAmounts(i-1);
end

x2 = zeros(N(2),1);
for j = 1:N(2);
    r = rand;
    while r > upperLimit || r < lowerLimit
        r = rand;
    end
    i = 2;
    while r > bucketValues(i)
        i = i + 1;
    end
    x2(j) = bucketAmounts(i-1);
end

x3 = zeros(N(3),1);
for j = 1:N(3);
    r = rand;
    while r > upperLimit || r < lowerLimit
        r = rand;
    end
    i = 2;
    while r > bucketValues(i)
        i = i + 1;
    end
    x3(j) = bucketAmounts(i-1);
end

x4 = zeros(N(4),1);
for j = 1:N(4);
    r = rand;
    while r > upperLimit || r < lowerLimit
        r = rand;
    end
    i = 2;
    while r > bucketValues(i)
        i = i + 1;
    end
    x4(j) = bucketAmounts(i-1);
end

figure;
bincounts = histc(x1, buckets);
bar(buckets,bincounts,'histc')
xlim([-5, 5])
title('N = 100 Samples')
xlabel('x'); ylabel('Number of samples');

figure;
bincounts = histc(x2, buckets);
bar(buckets,bincounts,'histc')
xlim([-5, 5])
title('N = 200 Samples')
xlabel('x'); ylabel('Number of samples');

figure;
bincounts = histc(x3, buckets);
bar(buckets,bincounts,'histc')
xlim([-5, 5])
title('N = 500 Samples')
xlabel('x'); ylabel('Number of samples');

figure;
bincounts = histc(x4, buckets);
bar(buckets,bincounts,'histc')
xlim([-5, 5])
title('N = 1000 Samples')
xlabel('x'); ylabel('Number of samples');
