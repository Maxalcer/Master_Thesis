from agent_n2 import Agent_N2_Nodes

agent = Agent_N2_Nodes(5, 10, 0.01, 0.2)

agent.train_net("../Data/5x10")

agent.save_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/n2/trained_net_no_cap.py")

agent.save_learning_curve("/home/mi/maxa55/Master_Thesis/Results/Learning Curves/n2/no_cap")

#agent.plot_learning_curve("learning_curve_noisy.png")

