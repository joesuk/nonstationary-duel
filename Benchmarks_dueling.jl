# Useful functions to create non-stationary dueling bandit instances, stored in matrix of mean preferences

# a stationary dueling bandit with prefference matrix P
function NoChange(T,P)
    K = length(P)
    Problem = zeros(K,K,T)
    for k in 1:K
        for j in 1:K
            Problem[k,j,:]=P[k,j]*ones(T)
        end
    end
    return Problem,zeros
end

# generate rewards with uniformly spaced breakpoints
function ProblemUnif(T,P)
    (K,K,Episodes)=size(P)
    nbBreaks = Episodes - 1
    Problem = zeros(K,K,T)
    part = round(Int,T/(nbBreaks+1)) # size of each episode
    # filling the matrix
    for c in 0:nbBreaks
	    for i in 1:K
                for j in 1:K
            Problem[i,j,(1+c*part):(c+1)*part]=Problem[i,j,(1+c*part):(c+1)*part].+P[i,j,c+1]
                end
            end
    end
    if (Episodes*part < T)
        for i in 1:K
            for j in 1:K
            Problem[i,j,(Episodes*part+1):T]=Problem[i,j,(Episodes*part+1):T].+P[i,j,Episodes]
            end
        end
    end
    BreakPoints = [i*part for i in 1:nbBreaks]
    return Problem,BreakPoints,winners
end

# generate SST & STI environment with fixed gap
function UnifSSTI(T,K,Episodes,eps,numWinners)
    # initialize
    P = zeros(K,K,T).+0.5 # preference matrix
    winners_full = zeros(Int,T) # list of winners
    part = round(Int,T/(Episodes)) # size of each episode
    initial_winners = Set(sample(collect(1:K), numWinners, replace = false)) # initial winners

    # filling the matrix    
    for c in 0:(Episodes-1)
        if c>0 # force a change in winner
            winners= Set(sample(setdiff!(collect(1:K),initial_winners), numWinners, replace = false))
        else
            winners=initial_winners
        end

        for i in 1:K
            for j in 1:K
                if (i in winners) && (j ∉ winners)
                    P[i,j,(1+c*part):(c+1)*part].=min(1,max(0,0.5 + eps))
                end
                P[j,i,(1+c*part):(c+1)*part] = -1*P[i,j,(1+c*part):(c+1)*part].+1
            end
        end
        winners_full[(1+c*part):(c+1)*part].=rand(winners) #winners[(c+1)]
    end
    return P, winners_full
end


# generate geometric BTL environment 
function geomBTL(T,K,Episodes,fixed_winner)
    # set up preference matrix and winners first
    P = zeros(K,K,T).+0.5
    winners_full = zeros(Int,T)
    part = round(Int,T/(Episodes))
    order=shuffle(collect(1:K))
    initial_winner = order[1]
    for c in 0:(Episodes-1)
        if c>0
            # choose random ordering of arms in each new episode
            order = shuffle(collect(1:K))
            ind = findfirst(isequal(initial_winner), order)
            if fixed_winner
                # possibly fix initial_winner as the winner
                last = order[K]
                order[K] = initial_winner
                order[ind] = last
            end
        end
        for i in 1:K
            for j in 1:K
                P[order[i],order[j],(1+c*part):(c+1)*part].=min(1,max(0,(1/2^(i))/((1/2^(i))+1/2^(j)) ) )
                P[order[j],order[i],(1+c*part):(c+1)*part] = -1*P[order[i],order[j],(1+c*part):(c+1)*part].+1
            end
        end
        winners_full[(1+c*part):(c+1)*part].=order[1]
    end
    return P, winners_full
end

# Gaussian random walk environment
function RandomWalk(T,K)
    P = zeros(K,K,T).+0.5
    for t=1:T
        for i in 1:K
            for j in (i+1):K
                if t==1
                    P[i,j,t]=rand(1)[1]
                else
                    P[i,j,t] = min(1,max(0,P[i,j,(t-1)] + 0.5 + rand(Normal(0,0.002))))
                end
                P[j,i,t] = 1-P[i,j,t]
            end
        end
    end
    return P
end


# compute dynamic Condorcet dueling regret
function ComputeCumRegretDueling(P,Choices,winners)
   # compute the vector of cumulative regret on a bandit instance based on the successive selections of an algorithm stored in Choices
   K,Kind,T = size(P)
   regret = 0
   Regret = zeros(Float32,1,T)
   for t in 1:T
       regret+=0.5*(P[winners[t],Choices[t,1],t] + P[winners[t],Choices[t,2],t] - 1)
      Regret[t]=regret
   end
   return Regret
end
