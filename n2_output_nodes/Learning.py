from agent import Agent

agent = Agent(5, 5, 0.01, 0.2)

agent.train_net("../Data", 128, 10)

agent.plot_learning_curve("learning_curve.png")

agent.save_net("trained_net.py")