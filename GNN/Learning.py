from agent_GNN_3 import Agent_GNN

agent = Agent_GNN(5, 10, 0.01, 0.2)

agent.train_net("../Data/5x10")

agent.save_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn/trained_net_3.py")

agent.save_learning_curve("/home/mi/maxa55/Master_Thesis/Results/Learning Curves/gnn/3")

#agent.plot_learning_curve("learning_curve_noisy.png")

