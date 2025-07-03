from agent import Agent

agent = Agent(5, 10, 0.01, 0.2)

agent.train_net("../Data/5x10", 64, 100)

agent.save_net("trained_net_noisy.py")

agent.save_learning_curve("noisy.npy")

#agent.plot_learning_curve("learning_curve_noisy.png")

