# FAQ

- The training does not converge, the loss goes up or stays up instead of going down
    > Please make sure to wait for a couple of epochs (~10). It is normal that strange things happen in the beginning. If that does not help, please reduce the learning rate by setting `param.HyperParameter.opt_param.lr`. You may try it in steps of decreasing the learning rate by a factor of 0.5.