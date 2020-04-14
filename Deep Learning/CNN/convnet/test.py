layers = [
      Conv((4, 4, 1, 20), strides=2, activation=lkrelu, filter_init=lambda shp: np.random.normal(size=shp)),
      Conv((5, 5, 20, 40), strides=2, activation=lkrelu, filter_init=lambda shp:  np.random.normal(size=shp)),
      Flatten((5, 5, 40)),
      FullyConnected((5*5*40, 100), activation=sigmoid, weight_init=lambda shp: np.random.normal(size=shp)),
      FullyConnected((100, 10), activation=linear, weight_init=lambda shp: np.random.normal(size=shp))
  ]
  net = Network(layers, lr=0.001, loss=cross_entropy)