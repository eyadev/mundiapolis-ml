import tensorflow.keras as keras                                                                                                    
def build_model(nx, layers, activations, lambtha, keep_prob):                                                                      
  model=keras.Sequential();                                                                                                      
  for i in range(len(layers)):                                                                                                
    model.add(keras.layers.Dense(layers[i],input_shape=(nx,),
                                 activation=activations[i],
                                 kernel_regularizer=keras.regularizers.L2(lambtha)))   