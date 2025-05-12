from agent import Agent

agent = Agent(5, 10, 0, 0)

agent.train_net("../Data/5x10", 128, 20)

agent.save_learning_curve("noiseless.npy")

agent.plot_learning_curve("learning_curve_noiseless.png")

agent.save_net("trained_net_noiseless.py")