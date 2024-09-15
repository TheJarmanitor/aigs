# Mini-project 1
## By Oleg Jarma Montoya

The final outcome of the mini project was to train a Reinforcement learning Agent able to succeed in the environment "Cartpole".
In order to achieve this, Deep Q networks were used.

### Network structure
A neural network with the following parameters was defined:
- input layer of size 4
- Hidden layers: 2
- Neurons per layer: 16
- Activation used in the hidden layers: ReLu
- output layer of size 2

Following this, an initialization was made using the "orthogonal" initializer, as recommended by read literature for DQN.

### Policy
The policy used was e-greedy

### Loss function
The loss function used was MSE. Using DDQN was considered for this part but was not explored.

### Training
For the training process the following methods and parameters were used.

- learning rate of 0.00005
- an initial epsilon of 0.5
- a discount factor of 0.9
- A replay buffer of size 1000 and a sample batch of size 32
- a "Soft update" with an update factor of function to make the target weights update
- An exponential epsilon decay function with a minimum epsilon value of 0.01
- a "Warmup phase". Where for 1000 steps fully random decisions would be made in order to fill the buffer with varied experiences
- A "best weights" function which analizes the average cumulative rewards every 1000 steps and saves the weights used if a new record was achieved.

### development
The biggest problem when working with DQN was the amount of hyperparameter tuning that is necessary to achieve above average results. Things like doing the wrong initialization or a high learning caused massive problems and subpar results.
The second problem, which could also be attributed to the previous one, was with local minimums and the lack of variety in the experiences done by the agent. This was "relatively" mitigated with the warmup phase.

### results.
An agent was trained with "better than random" capabilities. This can of course be improved in future tries.
Some possible improvement sugested by literature:
- Different Policies
- "Intelligent" buffer filtering
- Batch normalization of the samples
