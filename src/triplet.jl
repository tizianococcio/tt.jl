using Printf
using ProgressBars
using Distributions

const global SIM_VER = "0.1.6"

#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
function sim(weights::Matrix{Float64},
			popmembers::Matrix{Int64},
			spikes,#::SpikeTimit.FiringTimes,
			transcriptions::SpikeTimit.Transcriptions,
			net::LKD.NetParams,
			store::LKD.StoreParams,
			weights_params::LKD.WeightParams,
			projections::LKD.ProjectionParams,
			tri_stdp::TripletSTDP)

	@unpack dt, simulation_time, learning =	net
	@unpack folder, save_weights, save_states, save_network, save_timestep = store
	@unpack Ne, Ni = weights_params
	@unpack neurons, ft = spikes
	##labels and savepoints
	
	savepoints = SpikeTimit.get_savepoints(transcriptions, 
										per_word = store.points_per_word, 
										per_phone=store.points_per_phone)
	words = transcriptions.words.signs
	phones = transcriptions.phones.signs
	jex_input = projections.je

	#membrane dynamics
	taue = 20 #e membrane time constant
	taui = 20 #i membrane time constant
	vleake = -55 #e resting potential
	vleaki = -62 #i resting potential
	deltathe = 2 #eif slope parameter
	C = 300 #capacitance
	erev = 0 #e synapse reversal potential
	irev = -75 #i synapse reversal potntial
	vth0 = -52 #initial spike voltage threshold
	ath = 10 #increase in threshold post spike
	tauth = 30 #threshold decay timescale
	vre = -60 #reset potential
	taurefrac = 1 #absolute refractory period
	aw_adapt = 4 #adaptation parameter a
	bw_adapt = .805 #adaptation parameter b		# Should probably be 100-150 times bigger, according to Alessio and Hartmut
	tauw_adapt = 150 #adaptation timescale

	#connectivity
	Ncells = Ne+Ni
	tauerise = 1 #e synapse rise time
	tauedecay = 6 #e synapse decay time
	tauirise = .5 #i synapse rise time
	tauidecay = 2 #i synapse decay time
	rex = 1.5 #external input rate to e (khz)	(noise)
	rix = 0.9 #external input rate to i (khz)	(noise)

	jeemin = 1.78 #minimum ee strength
	jeemax = 21.4 #maximum ee strength

	jeimin = 48.7 #minimum ei strength
	jeimax = 243 #maximum ei strength

	# Synaptic weights
	jex = 0.78 #external to e strength	(noise)
	jix = 1.27 #external to i strength	(noise)

	#voltage based stdp
	altd = .0008 #ltd strength
	altp = .0014 #ltp strength
	thetaltd = -70 #ltd voltage threshold
	thetaltp = -49 #ltp voltage threshold
	tauu = 10 #time constant of low-pass filtered membrane voltage (for LTD)
	tauv = 7 #time constant of low-pass filtered membrane voltage (for LTP)
	taux = 15 #time constant low-pass filtered spike train

	#inhibitory stdp
	tauy = 20 #width of istdp curve
	eta = 1 #istdp learning rate
	r0 = .003 #target rate (khz)
    alpha = 2*r0*tauy; #rate trace threshold for istdp sign (kHz) (so the 2 has a unit)



	#populations
	Npop = size(popmembers,2) #number of assemblies
	Nmaxmembers = size(popmembers,1) #maximum number of neurons in a population

	#simulation
	Nskip = 1000 #how often (in number of timesteps) to save w_in
	vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
	normalize_time = 20 #how often to normalize rows of ee weights
	stdpdelay = 1000 #time before stdp is activated, to allow transients to die out

	times =Vector{Vector{Float64}}()
	for _ in 1:Ncells
		push!(times, Vector{Float64}())
	end

	forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
	forwardInputsI = zeros(Ncells)
	forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
	forwardInputsIPrev = zeros(Ncells)
	
	spiked = zeros(Bool,Ncells)

	xerise = zeros(Ncells) #auxiliary variables for E/I currents (difference of exponentials)
	xedecay = zeros(Ncells)
	xirise = zeros(Ncells)
	xidecay = zeros(Ncells)

	expdist = Exponential()

	v = zeros(Ncells) #membrane voltage
	nextx = zeros(Ncells) #time of next external excitatory input
	sumwee0 = zeros(Ne) #initial summed e weight, for normalization
	Nee = zeros(Int,Ne) #number of e->e inputs, for normalization
	rx = zeros(Ncells) #rate of external input

	for cc = 1:Ncells
		v[cc] = vre + (vth0-vre)*rand() # Compute menbrane voltage of neuron
		if cc <= Ne 					# Is the neuron an E neuron?
			rx[cc] = rex				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc]	# time of next external excitatory input becomes smaller if rate of external input is larger
			for dd = 1:Ne
				sumwee0[cc] += weights[dd,cc]
				if weights[dd,cc] > 0
					Nee[cc] += 1
				end
			end
		else							# In case of an I neuron
			rx[cc] = rix				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc] # time of next external excitatory input becomes smaller if rate of external input is larger
		end
	end
	

	vth = vth0*ones(Ncells) #adaptive threshold
	wadapt = aw_adapt*(vre-vleake)*ones(Ne) #adaptation current
	lastSpike = -100*ones(Ncells) #last time the neuron spiked
	trace_istdp = zeros(Ncells) #low-pass filtered spike train for istdp
	u_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	v_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	x_vstdp = zeros(Ne)	# spike train used in the voltage-based STDP rule (formula 5)

	Nsteps = round(Int,simulation_time/dt)
	inormalize = round(Int,normalize_time/dt)
	rates = zeros(Float32, 2, Nsteps)

	exc_spike_count_bin = 0
	inh_spike_count_bin = 0

	# This will assign the first firing time. Is set to -1 if all_ft contains no firing times
	next_firing_time = -1
	firing_index = 1
	if !isempty(ft)
		next_firing_time = ft[firing_index]
	end

	# these are to manage the saving of the states
	words_sp = savepoints.words
	phones_sp = savepoints.phones
	word_index = 1
	phone_index = 1
	inword_index = 1
	inphone_index = 1
	next_word_savepoint = words_sp[word_index, inword_index]
	next_phone_savepoint = phones_sp[phone_index, inphone_index]

	measurements_per_word = size(words_sp,2)
	measurements_per_phone = size(phones_sp,2)
	word_w = Matrix{Float64}(undef, Ne,measurements_per_word)
	phone_w =Matrix{Float64}(undef, Ne,measurements_per_phone)
	word_v = Matrix{Float64}(undef, Ne,measurements_per_word)
	phone_v =Matrix{Float64}(undef, Ne,measurements_per_phone)

	println("starting simulation v$SIM_VER")
	# trackers for 3 neurons
	#voltage_neuron_1_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	#adaptation_current_neuron_1_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	#adaptive_threshold = 0.0*Vector{Float64}(undef,Nsteps)
	voltage_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	adaptation_current_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	adaptive_threshold_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	
    nzRowsAll = [findall(weights[nn,1:Ne].!=0) for nn = 1:Ncells] #Dick: for all neurons lists E postsynaptic neurons
    nzColsEE  = [findall(weights[1:Ne,mm].!=0) for mm = 1:Ne]     #Dick: for E neurons lists E presynaptic neurons
    nzRowsEE  = [findall(weights[mm,1:Ne].!=0) for mm = 1:Ne]     #Dick: for E neurons lists E postsynaptic neurons
    nzColsIE  = [findall(weights[Ne+1:Ncells,mm].!=0).+Ne for mm = 1:Ne] #Dick: for E neurons lists I presynaptic neurons
    nzforEtoAll  = [findall(weights[nn,:].!=0) for nn = 1:Ne] #for E neurons lists All postsynaptic neurons
    nzforItoAll  = [findall(weights[nn,:].!=0) for nn = Ne+1:Ncells] #for I neurons lists All postsynaptic neurons

	# presynaptic detectors
	r1 = zeros(Ne)
	r2 = zeros(Ne)

	# postsynaptic detectors
	o1 = zeros(Ne)	
	o2 = zeros(Ne)

	# detectors trackers
	do1 = 0.0*Vector{Float64}(undef,Nsteps)
	do2 = 0.0*Vector{Float64}(undef,Nsteps)
	dr1 = 0.0*Vector{Float64}(undef,Nsteps)
	dr2 = 0.0*Vector{Float64}(undef,Nsteps)

	#begin main simulation loop
	iterations = ProgressBar(1:Nsteps)
	@fastmath @inbounds for tt = iterations
		
		t = dt*tt
		mob_mean = tt- 100 >1 ? tt-100 : 1
		set_multiline_postfix(iterations,string(@sprintf("Rates: %.2f %.2f, %2.f", mean(rates[:,mob_mean:tt], dims=2)..., t )))
        #excitatory synaptic normalization/scaling

        if mod(tt,inormalize) == 0
            for cc = 1:Ne
                sumwee = @views sum(weights[1:Ne,cc]) #sum of presynaptic weights

                #normalization:
                invsumwee = inv(sumwee)
                for dd in nzColsEE[cc]
                    weights[dd,cc] *= sumwee0[cc]*invsumwee
                end

                #enforce range
                for dd in nzColsEE[cc]
					(weights[dd,cc] < jeemin) && (weights[dd,cc] = jeemin)
					(weights[dd,cc] > jeemax) && (weights[dd,cc] = jeemax)
                end
            end
        end #end normalization

		# Add the external input signal into the model.
		if tt == next_firing_time
			firing_populations = neurons[firing_index]
			firing_index +=1
			while ft[firing_index] == next_firing_time
				append!(firing_populations, neurons[firing_index])
				firing_index +=1
			end
			# @show tt, firing_index, firing_populations
			for pop in firing_populations
				for member in popmembers[:,pop]
					if member > -1
						forwardInputsEPrev[member] += jex_input
					end
				end
			end
			if firing_index < length(ft)
				next_firing_time = ft[firing_index]
			end
		end
		
		fill!(forwardInputsE,0.)
		fill!(forwardInputsI,0.)
		fill!(spiked,false)

		#update single cells
		for cc = 1:Ncells
			trace_istdp[cc] -= dt*trace_istdp[cc]/tauy		# inhibitory synaptic plasticity (formula 6?)

			while(t > nextx[cc]) #external input
				nextx[cc] += rand(expdist)/rx[cc]
				if cc <= Ne
					forwardInputsEPrev[cc] += jex	# noise added to the excitatory neurons
				else
					forwardInputsEPrev[cc] += jix	# noise added to the inhibitory neurons
				end
			end

			# compute the increase in currents
			xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
			xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
			xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
			xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

			if cc <= Ne	# is the current cell an E neuron?
				vth[cc] += dt*(vth0 - vth[cc])/tauth;	# Adaptive threshold of E neurons (formula 2)
				wadapt[cc] += dt*(aw_adapt*(v[cc]-vleake) - wadapt[cc])/tauw_adapt;	# Adaptation current of E neurons (formula 3)
				# u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu;	# update membrane voltage
				# v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv;	# update membrane voltage
				# x_vstdp[cc] -= dt*x_vstdp[cc]/taux;	# update spike train
			end

			if t > (lastSpike[cc] + taurefrac) #not in refractory period
				# update membrane voltage
				ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);
				gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);

				if cc <= Ne #excitatory neuron (eif), has adaptation
					dv = (vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C - wadapt[cc]/C; # voltage dynamics, formula 1 (contains results from formulas 1 & 2)
					v[cc] += dt*dv;
					if v[cc] > vpeak	# if the voltage is higher than threshold, spike
						spiked[cc] = true
					end
				else
					dv = (vleaki - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;	# voltage dynamics, formula 1
					v[cc] += dt*dv;
					if v[cc] > vth0	# if the voltage is higher than threshold, spike
						spiked[cc] = true
					end
				end

				voltage_tracker[tt] = v[1]
				adaptation_current_tracker[tt] = wadapt[1]
				adaptive_threshold_tracker[tt] = vth[1]
				# do1[tt] = o1[1]
				# dr1[tt] = r1[1]
				# do2[tt] = o2[1]
				# dr2[tt] = r2[1]

				if spiked[cc] #spike occurred
					push!(times[cc], t);	# Times at which the neurons spiked
					v[cc] = vre;	# reset voltage of neuron to reset potential
					lastSpike[cc] = t; # last spike to occur was just now
                    trace_istdp[cc] += 1.0; #increase the spike trace

                    if cc <= Ne
                        # x_vstdp[cc] += 1.0/taux;
                        vth[cc] = vth0 + ath;
                        wadapt[cc] += bw_adapt
                    end

					#loop over synaptic projections
                    if cc <= Ne #excitatory neuron
                        for dd in nzforEtoAll[cc] #to all postsynaptic neurons
                            forwardInputsE[dd] += weights[cc,dd];
                        end
                    else #inhibitory neuron
                        for dd in nzforItoAll[cc - Ne] #to all postsynaptic neurons
                            forwardInputsI[dd] += weights[cc,dd];
                        end
                    end # if Exc or Inh
				end #end if(spiked)
			end #end if(not refractory)
		end
			
		if learning
			# run on pre-synaptic cells.
			for cc = 1:Ncells
				# istdp (formula 6)
				if spiked[cc] && (t > stdpdelay)
					if cc <= Ne                 # excitatory neuron fired, potentiate i inputs
						for dd in nzColsIE[cc]  # loop over postsynaptic nonzero synapses
							weights[dd,cc] += eta*trace_istdp[dd]
						(weights[dd,cc] > jeimax) && (weights[dd,cc] = jeimax);
						end
					else       # presynaptic inhibitory neuron fired, modify outputs to e neurons
						for dd in nzRowsAll[cc] # only loop over nonzero synapses
							weights[cc,dd] += eta*(trace_istdp[dd] - alpha)
							(weights[cc,dd] > jeimax) && (weights[cc,dd] = jeimax);
							(weights[cc,dd] < jeimin) && (weights[cc,dd] = jeimin);
						end
					end
				end # end istdp

				# triplet version 3
				if (t > stdpdelay) && (cc <= Ne)
					if spiked[cc]
						r1[cc] += 1
						o1[cc] += 1
						# presynaptic neuron fired
						for dd in nzRowsEE[cc]  # loop over postsynaptic
							# LTD
							weights[cc,dd] -= o1[dd] * (tri_stdp.A_minus_2 + tri_stdp.A_minus_3 * r2[cc]);
							(weights[cc,dd] < jeemin) && (weights[cc,dd] = jeemin);
						end
						# postsynaptic neuron fired
						for dd in nzColsEE[cc] # loop over presynaptic
							# LTP
							weights[dd,cc] += r1[dd] * (tri_stdp.A_plus_2 + tri_stdp.A_plus_3 * o2[cc]);
							(weights[dd,cc] > jeemax) && (weights[dd,cc] = jeemax);
						end
						r2[cc] += 1
						o2[cc] += 1
					end
					r1[cc] -= dt/tri_stdp.tau_plus * r1[cc]
					r2[cc] -= dt/tri_stdp.tau_x * r2[cc]
					o1[cc] -= dt/tri_stdp.tau_minus * o1[cc]
					o2[cc] -= dt/tri_stdp.tau_y * o2[cc]
				end

			end

		end #end loop over cells


		# the previous forward inputs of the next time step are the forward inputs of the current time step
		forwardInputsEPrev[:] .= forwardInputsE[:]
		forwardInputsIPrev[:] .= forwardInputsI[:]


		rates[1,tt] = mean(trace_istdp[1:Ne])/2/tauy*1000
		rates[2,tt] = mean(trace_istdp[Ne+1:end])/2/tauy*1000

		if (tt == 1 || mod(tt, save_timestep) == 0) && save_weights
			@time LKD.save_network_weights(weights, tt/1000, folder)
		end

		if save_states
			#Save the network state with the current label
			if tt == next_word_savepoint
				#Add current state to phone_adapt and phone_v
				word_w[:,inword_index] .= wadapt[:]
				word_v[:,inword_index] .= v[1:Ne]
				#Save to file if we are at the last measurement for the current phone
				if inword_index == measurements_per_word
					LKD.save_network_state(word_w, word_v, words[word_index], word_index, joinpath(folder,"word_states"))
					if word_index < length(words)
						word_index += 1
						inword_index = 1
					end
				else
					inword_index += 1
				end
				next_word_savepoint = words_sp[word_index, inword_index]
			end

			#If the current timestep is equal to the end of the currently presented phone
			#Save the network state with the current phone label
			if tt == next_phone_savepoint
				#Add current state to phone_adapt and phone_v
				phone_w[:,inphone_index] .= wadapt[:]
				phone_v[:,inphone_index] .= v[1:Ne]
				#Save to file if we are at the last measurement for the current phone
				if inphone_index == measurements_per_phone
					LKD.save_network_state(phone_w, phone_v, phones[phone_index], phone_index, joinpath(folder,"phone_states"))
					if phone_index < length(phones)
						phone_index += 1
						inphone_index = 1
					end
				else
					inphone_index += 1
				end
				next_phone_savepoint = phones_sp[phone_index, inphone_index]
			end
		end #saving states
	end #end loop over time
	
	if save_network
		@time LKD.save_network_weights(weights, simulation_time/1000, folder)
		@time LKD.save_network_spikes(times, folder)
		@time LKD.save_network_rates(rates, folder)	# Save mean weights over inhibitory neurons
		
		LKD.save_neuron_membrane(voltage_tracker, folder)
		LKD.save_neuron_membrane(adaptation_current_tracker, folder; type="w_adapt")
		LKD.save_neuron_membrane(adaptive_threshold_tracker, folder; type="adaptive_threshold")
		println("Done saving parameters")
	end

	return weights, times, rates, (voltage_tracker, adaptation_current_tracker, adaptive_threshold_tracker, dr1, do1, dr2, do2, (Matrix{Float64}(undef, 0,0), Matrix{Float64}(undef, 0,0)))
end
