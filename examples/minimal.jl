using tt
params = tt.LKD.InputParams(
    dialects=[1], 
    samples=1, 
    gender=['f'], 
    words=["that", "she", "all", "your", "me", "had", "like", "don't", "year", "water", "dark", "rag", "oily", "wash", "ask", "carry", "suit"], 
    repetitions=1, 
    shift_input=2, 
    encoding="bae"
)
wp = tt.LKD.WeightParams()
input = tt.InputLayer(params, wp, tt.TripletSTDP())
snn = tt.SNNLayer(input)
snn_trained = tt.train(snn) # STDP-adaptation
snn_test = tt.test(snn) # simulation without STDP
classification = tt.words_classifier(input.id, snn_test)
