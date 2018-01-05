import math

def call_bin_euro(S, K, T, n, r, s):
    #This function computes and returns the value of the corresponding European call option value using a binomial tree model

    n = int(n);                                 # convert n from float to int
    u = math.exp(s*((T/n)**0.5));               # set u using equation
    d = 1/u;                                    # set d using u
    p = (math.exp(r*T/n) - d)/ (u - d);         # obtain p using equation
    stockArray = [S];                           # initialize p, probability of success
    numNodes = int((n+1)*(n+2)/2);              # get the number of nodes need in the binomial tree based on n steps

    # Derive values of Stock Price for each step in the binomial tree and save in array S
    # The array is in this format: [S0, S1u, S1d, S2uu, S2ud, S2dd, S3uuu, S3uud, S2udd, S3ddd, ...] based on n
    # i index keeps track of current step, j index keeps track of current node within step
    # indexing of i and j is used to refer to parent node
    for i in range(1, n+1):
        for j in range(0, i):
            stockArray.append(u * stockArray[len(stockArray) - i ]);
        stockArray.append(d * stockArray[len(stockArray) - 1 - i]);

    # Initialize call value array and obtain American call value based on stock price and strike price at each node
    # Only the last step's values are needed for European option, but same procedure is used and values
    # are overwritten later.
    callArray = [];
    for i in range(0,len(stockArray)):
        if stockArray[i] > K:
            callArray.append(stockArray[i] - K)
        else:
            callArray.append(0)


    timeShift = math.exp(-r*T/n);               # discounting multiplier for each tree step
    currPtr = int(numNodes - 2 - n);            # current Node
    # Backtrack through call array and get discounted value at each step i and node within step j
    for i in range(n, 0, -1):
        for j in range(0, i):
            callArray[currPtr] = p * callArray[currPtr + i] + (1-p) * callArray[currPtr + i + 1];
            callArray[currPtr] = callArray[currPtr] * timeShift;
            currPtr = currPtr - 1;

    call = callArray[0];             # Assign call value as first item of call array
    return call



def put_bin_euro(S, K, T, n, r, s):
    #This function computes and returns the value of the corresponding European put option value using a binomial tree model

    n = int(n);                                 # convert n from float to int
    u = math.exp(s*((T/n)**0.5));               # set u using equation
    d = 1/u;                                    # set d using u
    p = (math.exp(r*T/n) - d)/ (u - d);         # obtain p using equation
    stockArray = [S];                           # initialize p, probability of success
    numNodes = int((n+1)*(n+2)/2);              # get the number of nodes need in the binomial tree based on n steps

    # Derive values of Stock Price for each step in the binomial tree and save in array S
    # The array is in this format: [S0, S1u, S1d, S2uu, S2ud, S2dd, S3uuu, S3uud, S2udd, S3ddd, ...] based on n
    # i index keeps track of current step, j index keeps track of current node within step
    # indexing of i and j is used to refer to parent node
    for i in range(1, n + 1):
        for j in range(0, i):
            stockArray.append(u * stockArray[len(stockArray) - i]);
        stockArray.append(d * stockArray[len(stockArray) - 1 - i]);

        # Initialize put value array and obtain American put value based on stock price and strike price at each node
        # Only the last step's values are needed for European option, but same procedure is used and values
        # are overwritten later.
        putArray = [];
    for i in range(0, len(stockArray)):
        if stockArray[i] < K:
            putArray.append(K - stockArray[i])
        else:
            putArray.append(0)

    timeShift = math.exp(-r * T / n);  # discounting multiplier for each tree step
    currPtr = int(numNodes - 2 - n);  # current Node
    # Backtrack through put array and get discounted value at each step i and node within step j
    for i in range(n, 0, -1):
        for j in range(0, i):
            putArray[currPtr] = p * putArray[currPtr + i] + (1 - p) * putArray[currPtr + i + 1];
            putArray[currPtr] = putArray[currPtr] * timeShift;
            currPtr = currPtr - 1;

    put = putArray[0];           # Assign put value as first item of put array
    return put



def call_bin_amer(S, K, T, n, r, s):
    #This function computes and returns the value of the corresponding American call option value using a binomial tree model

    n = int(n);                                 # convert n from float to int
    u = math.exp(s*((T/n)**0.5));               # set u using equation
    d = 1/u;                                    # set d using u
    p = (math.exp(r*T/n) - d)/ (u - d);         # obtain p using equation
    stockArray = [S];                           # initialize p, probability of success
    numNodes = int((n+1)*(n+2)/2);              # get the number of nodes need in the binomial tree based on n steps

    # Derive values of Stock Price for each step in the binomial tree and save in array S
    # The array is in this format: [S0, S1u, S1d, S2uu, S2ud, S2dd, S3uuu, S3uud, S2udd, S3ddd, ...] based on n
    # i index keeps track of current step, j index keeps track of current node within step
    # indexing of i and j is used to refer to parent node
    for i in range(1, n+1):
        for j in range(0, i):
            stockArray.append(u * stockArray[len(stockArray) - i ]);
        stockArray.append(d * stockArray[len(stockArray) - 1 - i]);

    # Initialize call value array and obtain American call value based on stock price and strike price at each node
    # These values will later be used to compare European option value with this American early exercise value
        callArray = [];
    for i in range(0,len(stockArray)):
        if stockArray[i] > K:
            callArray.append(stockArray[i] - K)
        else:
            callArray.append(0)

    timeShift = math.exp(-r*T/n);               # discounting multiplier for each tree step
    currPtr = int(numNodes - 2 - n);            # current Node
    # Backtrack through call array and get discounted value at each step i and node within step j and compare to
    # American early exerise value and assign the higher of the two
    for i in range(n, 0, -1):
        for j in range(0, i):
            euroTemp = p * callArray[currPtr + i] + (1-p) * callArray[currPtr + i + 1];
            euroTemp = euroTemp * timeShift;

            if euroTemp > callArray[currPtr]:
                callArray[currPtr] = euroTemp;
            currPtr = currPtr - 1;

    call = callArray[0];        # Assign call value as first item of call array
    return call



def put_bin_amer(S, K, T, n, r, s):
    #This function computes and returns the value of the corresponding American put option value using a binomial tree model

    n = int(n);                                 # convert n from float to int
    u = math.exp(s*((T/n)**0.5));               # set u using equation
    d = 1/u;                                    # set d using u
    p = (math.exp(r*T/n) - d)/ (u - d);         # obtain p using equation
    stockArray = [S];                           # initialize p, probability of success
    numNodes = int((n+1)*(n+2)/2);              # get the number of nodes need in the binomial tree based on n steps

    # Derive values of Stock Price for each step in the binomial tree and save in array S
    # The array is in this format: [S0, S1u, S1d, S2uu, S2ud, S2dd, S3uuu, S3uud, S2udd, S3ddd, ...] based on n
    # i index keeps track of current step, j index keeps track of current node within step
    # indexing of i and j is used to refer to parent node
    for i in range(1, n + 1):
        for j in range(0, i):
            stockArray.append(u * stockArray[len(stockArray) - i]);
        stockArray.append(d * stockArray[len(stockArray) - 1 - i]);

    # Initialize put value array and obtain American call value based on stock price and strike price at each node
    # These values will later be used to compare European option value with this American early exercise value
        putArray = [];
    for i in range(0, len(stockArray)):
        if stockArray[i] < K:
            putArray.append(K - stockArray[i])
        else:
            putArray.append(0)

    timeShift = math.exp(-r * T / n);  # discounting multiplier for each tree step
    currPtr = int(numNodes - 2 - n);  # current Node
    # Backtrack through put array and get discounted value at each step i and node within step j and compare to
    # American early exerise value and assign the higher of the two
    for i in range(n, 0, -1):
        for j in range(0, i):
            euroTemp = p * putArray[currPtr + i] + (1 - p) * putArray[currPtr + i + 1];
            euroTemp = euroTemp * timeShift;
            if euroTemp > putArray[currPtr]:
                putArray[currPtr] = euroTemp;
            currPtr = currPtr - 1;

    put = putArray[0];      # Assign put value as first item of put array
    return put