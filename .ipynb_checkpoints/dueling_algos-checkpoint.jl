# Non-stationary dueling bandit algorithms

# ANACONDA algorithm from Saha & Buening, 2023.
function ANACONDA(Table,params=[1,1,1],verbose="off")
    # set parameters
    C=params[1] # eviction threshold
    P=params[2] # replay probability multiplier
    maxp=params[3] # max replay probability
    
    # table contains a table of rewards for the K arms up to horizon T
    (K,K,T)=size(Table)
    
    # index of the episode
    episode = 1
    ChosenArms = zeros(Int,T,2)
    ChangePoints = []
    t = 0 # round timer

    # replay schedule parameters
    M = trunc(Int,round(log2(T))) # max log length of replay
    lengths = zeros(Int,M) # lengths of replays
    
    # set eviction thresholds
    thresholds = zeros(T)
    for s in 1:T
        thresholds[s] = C*(ℯ-1)*(K*sqrt((4*log2(T)+2*log2(K))*(s))+K^(2)*(4*log2(T)+2*log2(K)))
    end

    # storage of aggregate gaps and base algorithms
    SUMS = zeros(K,K,T) .- 0.5 # array containing S[a',a,s] = S[a',a,s,t_current]
    Bases = Stack{Vector}() # stack of base algorithms
    
    while (t < T)

        # set up replay schedule for this episode/stationary phase
        outcomes = zeros(Bool,M,T)
        for i in 1:M # length of replay
            for j in 1:T # start time of replay
                outcomes[i,j] = rand(Binomial(1,min(maxp,(P*1/sqrt((2^i)*(j))))))
            end
            lengths[i] = 2^i
        end

        # initialize master set of arms for episode
        Master_Arms = collect(1:K) # master arm set
        push!(Bases,[(t+1),(t+1),(T+1),collect(1:K)]) # push first base alg
        newEpisode = false # should we start a new episode?
        epStart = t+1

        # start the episode/stationary phase
        while (!newEpisode)&&(t < T)
            t=t+1
            # form the set of candidate arms
            recent = first(Bases)
            recent_start = recent[1]
            recent_last = recent[2]
            recent_end = recent[3]
            recent_arms = recent[4]
            Active = copy(recent_arms)
            
            # draw a random arm
            if length(Active)>1
             I,J = sample(Active,2,replace=false)
            else
            I = Active[1]
            J = Active[1]
            end
            ChosenArms[t,1]=I
            ChosenArms[t,2]=J
            
            # observe rewards, a.k.a. binary preferences
            rew = rand(Binomial(1,(Table[I,J,t])))
            
            # update gap estimates
            SUMS[I,J,epStart:t]= SUMS[I,J,epStart:t] .+ (rew)*(length(Active)^2)
            SUMS[J,I,epStart:t]= SUMS[J,I,epStart:t] .+ (1-rew)*(length(Active)^2)
            
            # update current active arm set
            if t>maximum([epStart,recent_start])
            
                if (J != I)
                    
                    inds = collect(0:trunc(Int,floor(log2(1 + t - recent_start))))
                    inds = recent_start .- 1 .+ 2 .^ inds
                    back_inds = t .- inds .+ 1 #.- (recent_start - 1)
            
                
        # update current base-alg active set
        if (rew>0) && (sum(SUMS[I,J,inds].- thresholds[back_inds]) > 0)
        
        # remove arm from active set
        deleteat!(Active, findfirst(isequal(J), Active))
        # remove from master set
        if J in Master_Arms
        deleteat!(Master_Arms, findfirst(isequal(J), Master_Arms))
        end
        elseif (rew==0) && (sum(SUMS[J,I,inds].- thresholds[back_inds]) > 0)
        # remove arm from active set
        deleteat!(Active, findfirst(isequal(I), Active))
        # remove from master set
        if I in Master_Arms
        deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
        end
        end
end

# update master arm set if not already edited
if (recent_start > epStart) && ((J in Master_Arms) || (I in Master_Arms))
    #inds_e = collect(epStart:(recent_start-1))
    #back_inds_e = collect((t-recent_start+2):(t-epStart+1)) #inds_e .- epStart .+ (t - recent_start+2)
    inds_e = collect(0:trunc(Int,floor(log2(1 + (t - epStart) - (t - recent_start + 1)))))
    inds_e = epStart .- 1 .+ 2 .^ inds_e
    back_inds_e = t .- inds_e .+ 1 # .- epStart .+ (t - recent_start+2)
    # update master arm set over data outside current active alg
    if (rew>0) && (J in Master_Arms) && (sum(SUMS[I,J,inds_e].- thresholds[back_inds_e]) > 0)
        deleteat!(Master_Arms, findfirst(isequal(J), Master_Arms))
   #elseif (rew==0) && (I in Master_Arms) && (sum(SUMS[J,I,epStart:(recent_start-1)].-reverse(thresholds[(t-recent_start+2):(t-epStart+1)])) > 0)
    elseif (rew==0) && (I in Master_Arms) && (sum(SUMS[J,I,inds_e].- thresholds[back_inds_e]) > 0)
        deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
   end
end
end


# perform restart tests
if (length(Master_Arms)==0)
episode+=1
ChangePoints=append!(ChangePoints,t)
if verbose=="on"
    print("ANACONDA detected a sig. change at t=$(t)\n\n")
end
newEpisode = true
break
else
cleanBases(Bases,Active,Master_Arms,t,epStart,outcomes,M,lengths,K,"ANACONDA")
end

if (t % 1000==0) && (verbose=="on")
    println("round $(t): $(length(Master_Arms)) arms left and current $(recent))")
end
    end
   end
   return ChosenArms,ChangePoints
end

# METASWIFT algorithm from Suk & Agarwal, 2023.
function METASWIFT(Table,params=[1,1,1,1],verbose="off",short="on")
    # set parameters
    C=params[1]
    canchor=params[2]
    P=params[3]
    maxp=params[4]
    
   # table contains a table of rewards for the K arms up to horizon T
   (K,K,T)=size(Table)
    
   # index of the episode
   episode = 1
   ChosenArms = zeros(Int,T,2)
   ChangePoints = []
   t = 0 # round timer

    # replay setup
   M = trunc(Int,round(log2(T)))
   outcomes = zeros(Bool,M,T)
   lengths = zeros(Int,M)

    # set eviction thresholds
    thresholds = zeros(T)
    thresholds_anchor = zeros(T)
    for s in 1:T
        #thresholds[s] = C*(sqrt(K*log2(T)*(s))+4*K*log2(T))
        #thresholds[s] = (ℯ-1)*(sqrt(K*(4*maximum([0,log2(T/C)])+maximum([0,log2(K/C)]))*(s))+K*(4*maximum([0,log2(T/C)])+maximum([0,log2(K/C)])))
        thresholds[s] = C*(ℯ-1)*(sqrt(K*(4*log2(T)+log2(K))*(s))+K*(4*log2(T)+log2(K)))
        thresholds_anchor[s] = canchor*(ℯ-1)*(sqrt(K*(4*log2(T)+log2(K))*(s))+K*(4*log2(T)+log2(K)))
    end

     # storage of aggregate gaps and base algorithms
  SUMS = zeros(K,T) .- 0.5 # array containing the S[anchor,a,s,t_current]
    SUMS_anchor = zeros(K,T) .- 0.5 # array containing the S[a,anchor,s,t_current]
  Bases = Stack{Vector}() # stack of base algorithms
    replayTracker = zeros(T) # track rounds spent in replay

    
   while (t < T)

        # set up replay schedule for this episode
       for i in 1:M # length of replay
            lengths[i] = 2^i
           for j in 1:T # start time of replay
               outcomes[i,j] = rand(Binomial(1,min(maxp,(P*1/sqrt((lengths[i])*(j))))))
            end
        end

      # initialize master set of arms for episode
      Master_Arms = collect(1:K) # master arm set
      push!(Bases,[(t+1),(t+1),(T+1),collect(1:K)]) # push first base alg
      newEpisode = false # should we start a new episode?
      epStart = t+1
      anchor = rand(collect(1:K))
      last_anchor_switch = t+1
        
      # start the episode
      while (!newEpisode)&&(t < T)
          t=t+1
         # form the set of candidate arms
         recent = first(Bases)
         recent_start = recent[1]
         recent_last = recent[2]
         recent_end = recent[3]
         recent_arms = recent[4]
            Active = copy(recent_arms)
            
         # draw a random arm
         if length(Active)>1
             I = rand(setdiff(Active,[anchor]))
        else
            I = rand(Active)
        end
         ChosenArms[t,1]=anchor
         ChosenArms[t,2]=I

        # observe rewards
        rew = rand(Binomial(1,(Table[anchor,I,t])))
            
         # update gap estimates
         SUMS[I,epStart:t] = SUMS[I,epStart:t] .+ (rew)*length(Active)
        SUMS_anchor[I,epStart:t] = SUMS_anchor[I,epStart:t] .+ (1-rew)*length(Active)

         # update current active arm set
         if (t>maximum([epStart,recent_start]))

            if (recent_start != epStart)
                    replayTracker[t] = 1
            end

            if (anchor != I)
            if short=="on"
                inds = collect(0:trunc(Int,floor(log2(1 + t - recent_start))))
                inds = recent_start .- 1 .+ 2 .^ inds
                back_inds = t .- inds .+ 1 #.- (recent_start - 1)
            else
                inds = collect(recent_start:t)
                back_inds = collect(1:(t-recent_start+1))
            end
                    
            # update current base-alg active set
            if (rew>0) && (I in Active) && (sum(SUMS[I,inds] .- thresholds[back_inds]) > 0)
                  # remove arm from active set
                     deleteat!(Active, findfirst(isequal(I), Active))
                     # remove from master set
                     if I in Master_Arms
                        deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
                        if verbose=="on"
                            println("$(t):"," ","$(length(Master_Arms))"," ","$(length(Active))")
                        end
                    end
            end

            # update master arm set over data outside current active alg
            #inds_e = collect(epStart:(recent_start-1))
            #back_inds_e = collect((t-recent_start+2):(t-epStart+1)) #inds_e .- epStart .+ (t - recent_start+2)
        if (rew>0) && (recent_start > epStart) && (I in Master_Arms)
                if short=="on"
                    inds_e = collect(0:trunc(Int,floor(log2(1 + (t - epStart) - (t - recent_start + 1)))))
                    inds_e = epStart .- 1 .+ 2 .^ inds_e
                    back_inds_e = t .- inds_e .+ 1 # .- epStart .+ (t - recent_start+2)
                else
                    inds_e = collect(epStart:(recent_start-1))
                    back_inds_e =  collect((t-recent_start+2):(t-epStart+1))
                end
                if (sum(SUMS[I,inds_e].-thresholds[back_inds_e]) > 0)
                    # remove arm from master
                    deleteat!(Master_Arms, findfirst(isequal(I), Master_Arms))
                    if verbose=="on"
                        println("$(t):"," ","$(length(Master_Arms))"," ","$(length(Active))","(earlier)")
                    end
                end
           end

            # eliminate anchor if possible
            # from master set
            elim_anchor = false
            if short=="on"
                inds_anchor = collect(0:trunc(Int,floor(log2(1 + t - last_anchor_switch))))
                inds_anchor = last_anchor_switch .- 1 .+ 2 .^ inds_anchor
                back_inds_anchor = t .- inds_anchor .+ 1 #.- (last_anchor_switch - 1)
            else
                inds_anchor = collect(last_anchor_switch:t)
                back_inds_anchor = collect(1:(t-last_anchor_switch+1))
            end
            if (rew==0) && (anchor in Active) && (last_anchor_switch >= recent_start) && (sum(SUMS_anchor[I,inds_anchor] .- thresholds[back_inds_anchor]) > 0)
                  # remove arm from active set
                     deleteat!(Active, findfirst(isequal(anchor), Active))
                     # remove from master set
                     if (anchor in Master_Arms)
                        deleteat!(Master_Arms, findfirst(isequal(anchor), Master_Arms))
                        if verbose=="on"
                            #println("$(t):"," ","$(length(Master_Arms))"," ","$(length(Active))"," (evicted anchor)")
                        end
                    end
                    anchor = I
                   last_anchor_switch = t
                   elim_anchor = true
            end

            # evict anchor from master set
            if (anchor in Master_Arms) && (last_anchor_switch >= epStart) && (sum(SUMS_anchor[I,inds_anchor].-thresholds[back_inds_anchor]) > 0)

                # remove anchor arm from master
                deleteat!(Master_Arms, findfirst(isequal(anchor), Master_Arms))
                if verbose=="on"
                    #println("$(t):"," ","$(length(Master_Arms))"," ","$(length(Active))","(evicted anchor from earlier)")
                end
                anchor = I
               last_anchor_switch = t
                elim_anchor = true
           end

           # update anchor if not evicted
            if canchor < T
            if short=="on"
                inds = collect(0:trunc(Int,floor(log2(1 + t - epStart))))
                inds = epStart .- 1 .+ 2 .^ inds
                back_inds = t .- inds .+ 1 #.- (recent_start - 1)
            else
                inds = collect(recent_start:t)
                back_inds = collect(1:(t-recent_start+1))
            end
            if (I in Master_Arms) && (length(Master_Arms)>2) && !(elim_anchor) && sum(SUMS_anchor[I,inds] .- thresholds_anchor[back_inds])>0
                   anchor = I
                   last_anchor_switch = t+1
                   #if (t % 100 == 0) && verbose=="on"
                       #println("switched anchor to $(I)")
                    #end
            end

            end
            end
        end

         # perform restart tests
         if (length(Master_Arms)==0)
               episode+=1
               ChangePoints=append!(ChangePoints,t)
               if verbose=="on"
                  print("METASWIFT detected a sig. change at t=$(t)\n\n")
               end
                newEpisode = true
                break
            end
         #end


    # throw error if anchor evicted
    #if anchor ∉ Master_Arms
      #anchor = rand(Master_Arms)
      #println("anchor was evicted at $(t)!?")
    #end


    cleanBases(Bases,Active,Master_Arms,t,epStart,outcomes,M,lengths,K,"METASWIFT")
    # check if 90% of arms are gone
    #if (length(Active)<K*0.9)
        #println("0.1 gone")
    #end

    if t % 10000==0 && verbose=="on"
        println("round $(t): $(length(Master_Arms)) master arms, $(length(Active)) active arms, $(length(Bases)) base-algs")
        #println(length(Bases))
        #println("$I")
  end
    end
   end
    #println(" $(sum(replayTracker)) spend in replay")
   return ChosenArms,ChangePoints #episode, collect(Iterators.product(a, b,c))

end

# clean stack of base algorithms in meta-algorithm
function cleanBases(Bases,Active,Master_Arms,t,epStart,outcomes,M,lengths,K,which)
        recent = first(Bases)
         # update last active round of current base alg
         recent[2] = t
         recent_end = recent[3]
         recent_arms = recent[4]

         # clean and update base algs (in Bases stack)
            clean=true
            prune_arms = copy(Active)
            if (length(Bases)>1) #&&(t>=recent_end)
                 while clean
                     recent = first(Bases)
                     recent_start = recent[1]
                     #println("at $(t), first started at $(recent_start)")
                     recent_end = recent[3]
                     recent_arms = recent[4]
                     prune_arms = intersect(prune_arms,recent_arms)
                     if (length(setdiff(Master_Arms,recent_arms))>0)
                         println("ERROR: master arm set not subset of recent armset")
                    end
                    if (t >= recent_end) || ((length(recent_arms)==length(Master_Arms)) & (recent_start != epStart))
                        pop!(Bases)
                    else
                        clean=false
                    end
            end
            #Bases = setdiff(Bases,delete_Base)
            #println("length of bases is $(length(Bases))")
        end

            recent = first(Bases)
            recent[4] = intersect(recent[4],prune_arms)

         # randomly add some new replay
         if (t>epStart)&&(length(Active)<K)
             #approxtime=argmin(abs.(lengths.-(t-epStart)))
             approxtime=t-epStart
                min_ind = 2^(trunc(Int,ceil(log2(2))))
            if which=="ANACONDA"
                min_ind = 2^(trunc(Int,ceil(log2(2))))
            end
             inds = [i for i in min_ind:M if outcomes[i,approxtime] > 0]
             append!(inds,0)
             if (maximum(inds) > 0)
                 m = maximum(inds)
                 push!(Bases,[(t),(t),(t+1+lengths[m]),collect(1:K)])
            end
        end

end

# play a random arm
function RANDDUEL(Table)
   (K,K,T)=size(Table)
   ChosenArms = zeros(Int,T,2)
   #ReceivedRewards = zeros(T)
  for t in 1:T
      I,J = rand(collect(1:K),2)
         rew = rand(Binomial(1,(Table[I,J,t])))
         ChosenArms[t,1]=I
         ChosenArms[t,2]=J
         #ReceivedRewards[t]=rew
   end
   return ChosenArms
end

# dueling EXP3.S
function EXP3SDuel(Table,nbreak=0,gamma=0.1,alpha=0.01)
   # table contains a table of rewards for the K arms up to horizon T
   (K,K,T)=size(Table)
   episode=1
   ChangePoints = []
   if (nbreak>0)
      # optimized gamma as a function of the nb of breakpoints
      gamma = min(1,sqrt(K*(nbreak*log(K*T)+exp(1))/((exp(1)-1)*T)))
      alpha = 1/T
   end
   Weights1=(1/K)*ones(Float32,K) # vector of weights / probability to sample
   Weights2=(1/K)*ones(Float32,K) # vector of weights / probability to sample
   ChosenArms = zeros(Int,T,2)
   #ReceivedRewards = zeros(T)
   for t in 1:T
      Probas1 = (1-gamma)*Weights1 .+ gamma/K
      Probas2 = (1-gamma)*Weights2 .+ gamma/K
      I = sampleDiscrete(Probas1)
      J = sampleDiscrete(Probas2)
      # get the reward
      rew1 = rand(Binomial(1,(Table[I,J,t])))
      rew2 = 1-rew1
      ChosenArms[t,1]=I
      ChosenArms[t,2]=J
      #ReceivedRewards[t]=rew1
      # update the weights
      normrew1 = rew1/Probas1[I]
      normrew2 = rew2/Probas2[I]
      bonus = (exp(1)*alpha/K)
      for k in 1:K
         if k!=I
            Weights1[k]=Weights1[k]+bonus
            Weights2[k]=Weights2[k]+bonus
         else
            Weights1[k]=(exp(gamma*normrew1/K))*Weights1[k]+bonus
            Weights2[k]=(exp(gamma*normrew2/K))*Weights2[k]+bonus
         end
      end
      # normalization step: storing the normalized weights and working with them leads to the same algorithm
      Weights1 = Normalize(Weights1)
      Weights2 = Normalize(Weights2)
   end
   return ChosenArms
end

# interleaved filtering
function IF(Table,params=[1],verbose="off")
    C = params[1]
   # table contains a table of rewards for the K arms up to horizon T
   (K,K,T)=size(Table)
   # index of the episode
   ChosenArms = zeros(Int,T,2)
   t = 0 # round timer
      vals = zeros(K,K)# array containing estimated preferences
    counts = zeros(K,K) # array containing counts of plays

      # initialize master set of arms for episode
      anchor = rand(collect(1:K))
      Active = deleteat!(collect(1:K), findfirst(isequal(anchor), collect(1:K)))
      while (t < T)
         # play all arms in active set
         t=t+1
        if length(Active)>0
             for I in Active
                 rew = rand(Binomial(1,(Table[anchor,I,t])))
                 ChosenArms[t,1]=anchor
                 ChosenArms[t,2]=I
                 counts[anchor,I] = counts[anchor,I] + 1
                 counts[I,anchor] = counts[I,anchor] + 1
                 vals[anchor,I] = (vals[anchor,I]*(counts[anchor,I]-1) + rew)/counts[anchor,I]
                 vals[I,anchor] = (vals[I,anchor]*(counts[I,anchor]-1) + (1-rew))/counts[I,anchor]
            end
        else # skip ahead to end if already converged to anchor
            if verbose=="true"
                println("converged at $(t)")
            end
             ChosenArms[t:T,1] .= anchor
             ChosenArms[t:T,2] .= anchor
            t=T
        end
        copyActive = copy(Active)
        if length(copyActive)>0
            for I in copyActive
                if (vals[anchor,I] - C*sqrt(log2(T*K^2)/counts[anchor,I]) > 0.5)
                    deleteat!(Active,findfirst(isequal(I),Active))
                    if verbose=="on"
                        println("$(I) evicted at time $(t)")
                    end
                end
            end
        end
        copyActive = copy(Active)
        if length(copyActive)>0
            for I in copyActive
                if (vals[I,anchor] - C*sqrt(log2(T*K^2)/counts[I,anchor]) > 0.5)
                    anchor = I
                    Active = deleteat!(Active, findfirst(isequal(I), Active))
                    vals = zeros(K,K) # array containing estimated preferences
                    counts = zeros(K,K) # array containing counts of plays
                    if verbose=="on"
                        println("anchor switched to $(I) at time $(t)")
                    end
                    break
                end
            end
        end
        if (t % 1000==0) && (length(Active)==0) && (verbose=="on")
            println("time $(t) and $(length(Active)) arms left")
        end
    end
   return ChosenArms
end

# tune algorithm's parameters
function tune(T,K,alg::Function,params,ep_tries,prob::Function,trials)
    current_min = T
    best_params = []
    i=0
    best_params = 0
    best_changes = 0
    for p in params
        regs = zeros(Float32,(length(ep_tries),trials))
        changepoints = zeros(Int,(length(ep_tries),trials))
        iter = collect(Iterators.product(collect(1:length(ep_tries)),collect(1:trials)))
        Threads.@threads for (j,n) in iter
            #for j in length(ep_tries)
                P_temp, winners_temp = prob(T,K,ep_tries[j])
                ChosenArms,ChangePoints = alg(P_temp,p)
                reg_temp = ComputeCumRegretDueling(P_temp,ChosenArms,winners_temp)
                regs[j,n] = reg_temp[T]
                changepoints[j,n] = length(ChangePoints)
        end
        reg = mean(regs)

        if reg < current_min
            current_min = reg
            best_params = p
            best_changes = mean(changepoints)
        end

        if i % 10==0
            println("done $(i)/$(length(params)); regret $(current_min); params $(best_params); changes $(best_changes)")
        end

        #if mean(changepoints) > ep_tries[1]/2
        #    println("$(params) restarts $(mean(changepoints)) times with reg $(reg)")
        #end

        i = i + 1
    end
    return best_params, current_min, best_changes
end
