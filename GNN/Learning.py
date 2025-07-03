from agent_GNN import Agent_GNN

agent = Agent_GNN(5, 10, 0.01, 0.2)

agent.train_net("../Data/5x10")

agent.save_net("/home/mi/maxa55/Master_Thesis/Results/Trained Networks/gnn/trained_net_features.py")

agent.save_learning_curve("/home/mi/maxa55/Master_Thesis/Results/Learning Curves/gnn/features")

#agent.plot_learning_curve("learning_curve_noisy.png")

